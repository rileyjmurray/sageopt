"""
   Copyright 2020 Riley John Murray

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import unittest
import numpy as np
from sageopt.coniclifts import utilities as u


class TestUtilities(unittest.TestCase):

    def test_kernel_basis(self):
        np.random.seed(0)
        a = np.random.randn(7, 5)
        # single column
        a[:, 1] = np.zeros(7)
        b1 = u.kernel_basis(a)
        self.assertEqual(b1.shape, (5, 1))
        e1 = np.zeros(shape=(5, 1))
        e1[1, 0] = 1.0 * np.sign(b1[1, 0])
        self.assertLessEqual(np.linalg.norm(e1 - b1), 1e-8)
        # multi-column
        a[:, 4] = np.zeros(7)
        b2 = u.kernel_basis(a)  # 2nd and 5th standard basis vectors
        range_ker = b2[[1, 4], :]
        self.assertGreaterEqual(np.abs(np.linalg.det(range_ker)), 0.1)
        null_ker = b2[[0,2,3], :]
        self.assertLessEqual(np.linalg.norm(null_ker), 1e-8)
        # fully random
        a = np.random.randn(3, 10)
        b3 = u.kernel_basis(a)
        self.assertEqual(b3.shape, (10, 7))
        a_on_ker = a @ b3
        self.assertLessEqual(np.linalg.norm(a_on_ker), 1e-7)
        # a full-rank matrix
        a = np.eye(3)
        b4 = u.kernel_basis(a)
        self.assertEqual(b4.shape, (3, 0))
        pass
