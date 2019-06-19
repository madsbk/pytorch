#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/AccumulateType.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCAtomics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>

namespace at {
namespace native {

namespace hej {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int NROWS_PER_THREAD = 10;

#ifdef __HIP_PLATFORM_HCC__
    constexpr int WARP_SIZE = 64;
#else
    constexpr int WARP_SIZE = 32;
#endif


__global__
void segment_sizes_kernel(int64_t *ret, const int64_t *segment_offsets,
                          int64_t num_of_segments, int64_t blocksize, int64_t numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    const int64_t idx_start = segment_offsets[id];
    const int64_t idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];
    const int64_t size = idx_end - idx_start;
    ret[id] = (size + blocksize - 1) / blocksize;
  }
}

__global__
void split_segment_offsets_kernel(
        int64_t *ret,
        const int64_t *segment_sizes,
        const int64_t *segment_sizes_offsets,
        const int64_t *segment_offsets,
        int64_t num_of_segments,
        int64_t blocksize) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    int64_t idx = segment_sizes_offsets[id];
    const int64_t segment_size = segment_sizes[id];
    const int64_t segment_offset = segment_offsets[id];
    for (int64_t i=0; i<segment_size; ++i) {
      ret[idx++] = segment_offset + i * blocksize;
    }
  }
}


// This kernel assumes that all input tensors are contiguous.
template <typename scalar_t>
__global__ void compute_grad_weight(
    int64_t *indices, scalar_t *gradOutput,
    int64_t *offset2bag, int64_t *count, ptrdiff_t numel,
    int64_t stride, int mode_mean, const int64_t *bag_size,
    scalar_t* per_sample_weights, int64_t per_sample_weights_stride,
    int64_t* segment_offsets, int64_t num_of_segments, scalar_t *grad_weight_per_segment,
    const int64_t stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const int origRow = indices[idx];
    const int seq_number = offset2bag[origRow];
    const int gradOutputRow = seq_number * stride;

    acc_type<scalar_t, true> scale = count ? 1.0 / count[idx] : 1.0;
    if (per_sample_weights) {
      scale *= per_sample_weights[origRow * per_sample_weights_stride];
    }

    acc_type<scalar_t, true> gradient = gradOutput[gradOutputRow + startFeature];
    if (mode_mean) {
      gradient /= bag_size[seq_number];
    }
    weight += gradient * scale;
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

// This kernel assumes that all input tensors are contiguous.
template <typename scalar_t>
__global__ void sum_and_scatter(
    int64_t *input, scalar_t *gradWeight, int64_t stride,
    int64_t* segment_offsets, int64_t num_of_segments, const scalar_t *grad_weight_per_segment,
    const int64_t *segment_sizes_offsets, int64_t num_of_split_segments,
    const int64_t stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_sizes_offsets[id];
  const int idx_end = (id == num_of_segments-1)?num_of_split_segments:segment_sizes_offsets[id+1];
  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    weight += grad_weight_per_segment[idx*stride + startFeature];
  }
  const int weightRow = input[segment_offsets[id]] * stride;
  gradWeight[weightRow + startFeature] = weight;
}

} // anon namespace


Tensor embedding_dense_backward_cuda(
        const Tensor &grad,
        const Tensor &orig_indices,
        const Tensor &sorted_indices,
        const Tensor &offset2bag,
        const Tensor &bag_size,
        const Tensor &count,
        int64_t num_weights,
        bool scale_grad_by_freq,
        bool mode_mean,
        const Tensor& per_sample_weights) {

  auto stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  const ptrdiff_t numel = sorted_indices.numel();

  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  thrust::device_vector<int64_t> segment_offsets(numel);
  int64_t num_of_segments;
  {
    auto sorted_indices_dev = thrust::device_ptr<int64_t>(sorted_indices.data<int64_t>());
    auto dummy = at::empty_like(sorted_indices);
    auto dummy_dev = thrust::device_ptr<int64_t>(dummy.data<int64_t>());
    auto ends = thrust::unique_by_key_copy(
            policy,
            sorted_indices_dev,
            sorted_indices_dev + numel,
            thrust::make_counting_iterator(0),
            dummy_dev,
            thrust::raw_pointer_cast(segment_offsets.data()));
    num_of_segments = thrust::get<0>(ends) - dummy_dev;
  }

  thrust::device_vector<int64_t> segment_sizes(num_of_segments);
  {
    segment_sizes_kernel<<<THCCeilDiv(num_of_segments, (ptrdiff_t)32), 32, 0, stream>>> (
            thrust::raw_pointer_cast(segment_sizes.data()),
                    thrust::raw_pointer_cast(segment_offsets.data()),
                    num_of_segments,
                    NROWS_PER_THREAD,
                    numel);
  }
  thrust::device_vector<int64_t> segment_sizes_offsets(num_of_segments);
  thrust::exclusive_scan(
          policy,
          segment_sizes.begin(),
          segment_sizes.end(),
          segment_sizes_offsets.begin());

  int64_t num_of_split_segments = segment_sizes[num_of_segments-1] + segment_sizes_offsets[num_of_segments-1];
  thrust::device_vector<int64_t> split_segment_offsets(num_of_split_segments);
  {
    split_segment_offsets_kernel<<<THCCeilDiv(num_of_segments, (ptrdiff_t)32), 32, 0, stream>>> (
            thrust::raw_pointer_cast(split_segment_offsets.data()),
                    thrust::raw_pointer_cast(segment_sizes.data()),
                    thrust::raw_pointer_cast(segment_sizes_offsets.data()),
                    thrust::raw_pointer_cast(segment_offsets.data()),
                    num_of_segments,
                    NROWS_PER_THREAD);
  }

  auto grad_weight_per_segment = at::empty({num_of_split_segments, stride}, grad.options());
  const int stride_warped = THCCeilDiv(stride, (ptrdiff_t)WARP_SIZE)*WARP_SIZE;
  const int block = std::min(stride_warped, MAX_BLOCK_SIZE);
  const int grid = THCCeilDiv(num_of_split_segments*stride_warped, (ptrdiff_t)block);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          grad.scalar_type(), "embedding_bag_backward_cuda_compute_grad_weight", [&] {
            compute_grad_weight<
            scalar_t><<<grid, block, 0, stream>>>(
                  orig_indices.data<int64_t>(),
                          grad.data<scalar_t>(),
                          offset2bag.data<int64_t>(),
                          count.defined() ? count.data<int64_t>() : nullptr, numel, stride,
                          mode_mean, bag_size.data<int64_t>(),
                          per_sample_weights.defined() ? per_sample_weights.data<scalar_t>() : NULL,
                          per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,
                          thrust::raw_pointer_cast(split_segment_offsets.data()),
                          num_of_split_segments, grad_weight_per_segment.data<scalar_t>(), stride_warped);
          });
  THCudaCheck(cudaGetLastError());

  const int grid2 = THCCeilDiv(num_of_segments*stride_warped, (ptrdiff_t)block);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          grad.scalar_type(), "embedding_bag_backward_cuda_sum_and_scatter", [&] {
            sum_and_scatter<
            scalar_t><<<grid2, block, 0, stream>>>(
                  sorted_indices.data<int64_t>(),
                          grad_weight.data<scalar_t>(),
                          stride,
                          thrust::raw_pointer_cast(segment_offsets.data()),
                          num_of_segments, grad_weight_per_segment.data<scalar_t>(),
                          thrust::raw_pointer_cast(segment_sizes_offsets.data()), num_of_split_segments, stride_warped);
          });
  THCudaCheck(cudaGetLastError());
  return grad_weight;
}


}}