"""
   Copyright 2019 Riley John Murray

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
from sageopt.coniclifts.operators import affine
from sageopt.coniclifts.base import Variable
from sageopt.coniclifts.compilers import compile_constrained_system
from sageopt.coniclifts.cones import Cone


def var_maps_equal(vm1, vm2):
    if not len(vm1) == len(vm2):
        return False
    else:
        for v in vm1:
            if vm1[v].shape != vm2[v].shape:
                return False
            if np.any(vm1[v] != vm2[v]):
                return False
        return True


class TestCompilers(unittest.TestCase):

    def test_LP_systems(self):
        n = 4
        m = 5
        x = Variable(shape=(n, 1), name='x')

        G = np.random.randn(m, n)
        h = G @ np.abs(np.random.randn(n, 1))
        constraints = [G @ x == h,
                       x >= 0]
        # Reference case : the constraints are over x, and we are interested in no variables other than x.
        A0, b0, K0, var_mapping0, _, _ = compile_constrained_system(constraints)
        A0 = A0.toarray()
        #   A0 should be the (m+n)-by-n matrix formed by stacking -G on top of the identity.
        #   b0 should be the (m+n)-length vector formed by concatenating h with the zero vector.
        #   Should see K0 == [('0',m), ('+',n)]
        #   var_mapping0 should be a length-1 dictionary with var_mapping0['x'] == np.arange(n).reshape((n,1)).
        assert np.all(b0 == np.hstack([h.ravel(), np.zeros(shape=(n,))]))
        assert K0 == [Cone('0', m), Cone('+', n)]
        assert var_maps_equal(var_mapping0, {'x': np.arange(0, n).reshape((n, 1))})
        expected_A0 = np.vstack((-G, np.eye(n)))
        assert np.all(A0 == expected_A0)

    def test_SDP_system_1(self):
        x = Variable(shape=(2, 2), name='x', var_properties=['symmetric'])

        D = np.random.randn(2, 2)
        D += D.T
        D /= 2.0
        B = np.random.rand(1, 2)
        C = np.diag([3, 0.5])
        constraints = [affine.trace(D @ x) == 5,
                       B @ x @ B.T >= 1,
                       B @ x @ B.T >> 1,  # a 1-by-1 LMI
                       C @ x @ C.T >> -2]
        A, b, K, _, _, _ = compile_constrained_system(constraints)
        A = A.toarray()
        assert K == [Cone('0', 1), Cone('+', 1), Cone('P', 1), Cone('P', 3)]
        expect_row_0 = -np.array([D[0, 0], 2 * D[1, 0], D[1, 1]])
        assert np.allclose(expect_row_0, A[0, :])
        temp = B.T @ B
        expect_row_1and2 = np.array([temp[0, 0], 2 * temp[0, 1], temp[1, 1]])
        assert np.allclose(expect_row_1and2, A[1, :])
        assert np.allclose(expect_row_1and2, A[2, :])
        expect_rows_3to6 = np.diag([C[0, 0] ** 2, C[0, 0] * C[1, 1], C[1, 1] ** 2])
        assert np.allclose(expect_rows_3to6, A[3:, :])
        assert np.all(b == np.array([5, -1, -1, 2, 2, 2]))

