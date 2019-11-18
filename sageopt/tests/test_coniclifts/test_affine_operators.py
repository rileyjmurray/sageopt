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
from sageopt.coniclifts.base import Variable, Expression
from sageopt.coniclifts.operators import affine as aff


class TestAffineOperators(unittest.TestCase):

    def test_dot(self):
        x = Variable(shape=(4,))
        a = np.array([1, 2, 3, 4])
        expr0 = aff.dot(x, a)
        expr1 = aff.dot(a, x)
        x0 = np.random.rand(4).round(decimals=4)
        expect = np.dot(a, x0)
        x.value = x0
        actual0 = expr0.value
        actual1 = expr1.value
        assert actual0 == expect
        assert actual1 == expect
        assert Expression.are_equivalent(expr0, expr1)

    def test_multi_dot(self):
        A = np.random.rand(5, 3).round(decimals=3)
        X = Variable(shape=(3, 3), var_properties=['symmetric'])
        X0 = np.random.rand(3, 3).round(decimals=3)
        X0 += X0.T
        X.value = X0
        B = np.random.rand(3, 3).round(decimals=3)
        C = np.random.rand(3, 7).round(decimals=3)

        chain1 = [A, X, B, C]
        expr1 = aff.multi_dot(chain1)
        expect1 = np.linalg.multi_dot([A, X0, B, C])
        actual1 = expr1.value
        assert np.allclose(expect1, actual1)

        chain2 = [A, B, X, C]
        expr2 = aff.multi_dot(chain2)
        expect2 = np.linalg.multi_dot([A, B, X0, C])
        actual2 = expr2.value
        assert np.allclose(expect2, actual2)

    def test_inner(self):
        # test with scalar inputs
        x = Variable()
        a = 2.0
        expr0 = aff.inner(a, x)
        expr1 = aff.inner(x, a)
        assert Expression.are_equivalent(expr0, expr1)
        # test with multidimensional arrays
        a = np.arange(24).reshape((2, 3, 4))
        x = Variable(shape=(4,))
        x.value = np.arange(4)
        expr = aff.inner(a, x)
        expect = np.inner(a, np.arange(4))
        actual = expr.value
        assert np.allclose(expect, actual)

    def test_outer(self):
        x = Variable(shape=(3,))
        x0 = np.random.randn(3).round(decimals=3)
        x.value = x0
        a = np.array([1, 2, 3, 4])
        expr0 = aff.outer(a, x)
        assert np.allclose(expr0.value, np.outer(a, x0))
        expr1 = aff.outer(x, a)
        assert np.allclose(expr1.value, np.outer(x0, a))
        b = np.array([9, 8])
        expr2 = aff.outer(b, x)
        assert np.allclose(expr2.value, np.outer(b, x0))
        expr3 = aff.outer(x, b)
        assert np.allclose(expr3.value, np.outer(x0, b))

    def test_kron(self):
        I = np.eye(2)
        X = Variable(shape=(2, 2))
        expr0 = aff.kron(I, X)
        expr1 = aff.kron(X, I)
        X0 = np.random.randn(2, 2).round(decimals=3)
        X.value = X0
        assert np.allclose(expr0.value, np.kron(I, X0))
        assert np.allclose(expr1.value, np.kron(X0, I))

    def test_trace_and_diag(self):
        x = Variable(shape=(5,))
        A = np.random.randn(5, 5).round(decimals=3)
        for i in range(5):
            A[i, i] = 0
        temp = A + aff.diag(x)
        expr0 = aff.trace(temp)
        expr1 = aff.sum(x)
        assert Expression.are_equivalent(expr0, expr1)
