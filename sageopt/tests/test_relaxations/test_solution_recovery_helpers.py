"""
   Copyright 2018 Riley John Murray

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
from sageopt.relaxations.poly_solution_recovery import mod2linsolve, mod2rref


# noinspection SpellCheckingInspection
class TestPolySolutionRecoveryHelpers(unittest.TestCase):

    def test_mod2linsolve_1(self):
        n = 5
        A = np.ones(shape=(n, n))
        A = A - np.eye(n)
        A = A.astype(int)
        b = np.zeros(shape=(n, 1)).astype(int)
        x = mod2linsolve(A, b)
        # valid values are x == 0, and x == 1.
        assert np.all(x == 0) or np.all(x == 1)
        b = np.array([1, 1, 1, 1, 0]).astype(int)
        x = mod2linsolve(A, b)
        num_zeros = np.count_nonzero(x == 0)
        assert num_zeros == 1 or num_zeros == 4
        b = np.array([1, 1, 1, 1, 1]).astype(int)
        x = mod2linsolve(A, b)
        assert x is None

    def test_mod2rref_1(self):
        n = 5
        A = np.ones(shape=(n, n)) - np.eye(n)
        # the vector of all ones is in the nullspace of A.
        # we will concatenate the vector of all ones as a column of A,
        # to create a matrix "A1" whose final column should be a pivot column.
        A1 = np.hstack((A, np.ones(shape=(n, 1)))).astype(int)
        a1rref, p1 = mod2rref(A1)
        assert p1[-1] == n



