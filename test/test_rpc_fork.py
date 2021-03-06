#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from rpc_test import RpcTest
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


class RpcTestWithFork(MultiProcessTestCase, RpcTest):

    def setUp(self):
        super(RpcTestWithFork, self).setUp()
        self._fork_processes()

if __name__ == '__main__':
    run_tests()
