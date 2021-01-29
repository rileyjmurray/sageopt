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

    def test_block(self):
        A = np.eye(2) * 2
        A_cl = Variable(shape=A.shape)
        A_cl.value = A
        B = np.eye(3) * 3
        expected = np.block([
            [A, np.zeros((2, 3))],
            [np.ones((3, 2)), B]
        ])
        actual = aff.block([
            [A_cl, np.zeros((2, 3))],
            [np.ones((3, 2)), B]
        ])
        assert np.allclose(expected, actual.value)
        A_cl.value = 1 + A  # or 2*A, or really anything different from A itself.
        assert not np.allclose(expected, actual.value)

    def test_concatenate(self):
        a = np.array([[1, 2], [3, 4]])
        a_cl = Variable(shape=a.shape)
        a_cl.value = a
        b = np.array([[5, 6]])
        b_cl = Variable(shape=b.shape)
        b_cl.value = b
        expected = np.concatenate((a, b))
        actual = aff.concatenate((a_cl, b_cl))
        assert np.allclose(expected, actual.value)
        expected1 = np.concatenate((b, a))
        actual1 = aff.concatenate((b_cl, a_cl))
        assert np.allclose(expected1, actual1.value)

        a_cl.value = 1 + a
        assert not np.allclose(expected, actual.value)
        assert not np.allclose(expected1, actual1.value)

    def test_stack(self):
        array = np.random.randn(3, 4)
        arrays = [array for _ in range(10)]
        a_cl = Variable(shape=array.shape)
        a_cl.value = array
        arrays_cl = [a_cl for _ in range(10)]
        expected = np.stack(arrays, axis=0)
        actual = aff.stack(arrays_cl, axis=0)
        assert np.allclose(expected, actual.value)
        assert Expression.are_equivalent(expected.shape, actual.shape)

        expected1 = np.stack(arrays, axis=1)
        actual1 = aff.stack(arrays_cl, axis=1)
        assert np.allclose(expected1, actual1.value)
        assert Expression.are_equivalent(expected.shape, actual.shape)

        a_cl.value = 1 + array
        assert not np.allclose(expected, actual.value)
        assert not np.allclose(expected1, actual1.value)

    def test_column_stack(self):
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        a_cl = Variable(shape=a.shape)
        a_cl.value = a
        b_cl = Variable(shape=b.shape)
        b_cl.value = b
        expected = np.column_stack((a, b))
        actual = aff.column_stack((a_cl, b_cl))
        assert np.allclose(expected, actual.value)
        a_cl.value = 1 + a
        assert not np.allclose(expected, actual.value)

    def test_dstack(self):
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        a_cl = Variable(shape=a.shape)
        a_cl.value = a
        b_cl = Variable(shape=b.shape)
        b_cl.value = b
        expected = np.dstack((a, b))
        actual = aff.dstack((a_cl, b_cl))
        assert np.allclose(expected, actual.value) 
        c = np.array([[1], [2], [3]])
        d = np.array([[2], [3], [4]])
        expected1 = np.dstack((c, d))
        c_cl = Variable(shape=c.shape)
        c_cl.value = c
        d_cl = Variable(shape=d.shape)
        d_cl.value = d
        actual1 = aff.dstack((c_cl, d_cl))
        assert np.allclose(expected1, actual1.value)

        a_cl.value = 1 + a
        assert not np.allclose(expected, actual.value)
        c_cl.value = 1 + c
        assert not np.allclose(expected1, actual1.value)

    def test_split(self):
        x = np.arange(9.0)
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.split(x, 3)
        actual = aff.split(x_cl, 3)
        assert np.all([np.allclose(expected[i], actual[i].value) for i in range(3)])
        x_cl.value = 1 + x
        assert not np.all([np.allclose(expected[i], actual[i].value) for i in range(3)])

    def test_hsplit(self):
        x = np.arange(16.0).reshape(4, 4)
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.hsplit(x, 2)
        actual = aff.hsplit(x_cl, 2)
        assert np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])
        x_cl.value = 1 + x
        assert not np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])

    def test_vsplit(self):
        x = np.arange(16.0).reshape(4, 4)
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.vsplit(x, 2)
        actual = aff.vsplit(x_cl, 2)
        assert np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])
        x_cl.value = 1 + x
        assert not np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])

    def test_dsplit(self):
        x = np.arange(16.0).reshape(2, 2, 4)
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.dsplit(x, 2)
        actual = aff.dsplit(x_cl, 2)
        assert np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])
        x_cl.value = 1 + x
        assert not np.all([np.allclose(expected[i], actual[i].value) for i in range(2)])

    def test_array_split(self):
        x = np.arange(8.0)
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.array_split(x, 3)
        actual = aff.array_split(x_cl, 3)
        assert np.all([np.allclose(expected[i], actual[i].value) for i in range(3)])
        x_cl.value = 1 + x
        assert not np.all([np.allclose(expected[i], actual[i].value) for i in range(3)])

    def test_tile(self):
        x = np.array([0, 1, 2])
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        A = aff.tile(x_cl, 2)
        assert np.allclose(np.tile(x, 2), A.value)
        expr0 = aff.sum(A)
        expr1 = aff.sum(x) * 2
        assert Expression.are_equivalent(expr0.value, expr1)

    def test_repeat(self):
        x = np.array([3])
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        A = aff.repeat(x_cl, 4)
        assert np.allclose(np.repeat(x, 4), A.value)
        expr0 = aff.sum(A.value)
        expr1 = aff.sum(x_cl) * 4
        assert Expression.are_equivalent(expr0, expr1.value)
        # x_cl.value = x + 1
        # assert not np.allclose(np.repeat(x, 4), A)

        x1 = np.array([[1, 2], [3, 4]])
        x1_cl = Variable(shape=x1.shape)
        x1_cl.value = x1
        A1 = aff.repeat(x1_cl, 2)
        assert np.allclose(np.repeat(x1, 2), A1.value)
        expr2 = aff.sum(A1.value)
        expr3 = aff.sum(x1_cl) * 2
        assert Expression.are_equivalent(expr2, expr3.value)
        
    def test_diagflat(self):
        x = np.array([[1, 2], [3, 4]])
        x_cl = Variable(shape=x.shape)
        x_cl.value = x
        expected = np.diagflat(x)
        actual = aff.diagflat(x_cl)
        assert np.allclose(expected, actual.value)

    def test_tril(self):
        A = np.random.randn(5, 5).round(decimals=3)
        A_cl = Variable(shape=A.shape)
        A_cl.value = A
        temp = aff.tril(A)
        expr0 = aff.sum(temp)
        expr1 = aff.sum(np.tril(A_cl))
        assert Expression.are_equivalent(expr0, expr1.value)

    def test_triu(self):
        A = np.random.randn(5, 5).round(decimals=3)
        A_cl = Variable(shape=A.shape)
        A_cl.value = A
        temp = aff.triu(A)
        expr0 = aff.sum(temp)
        expr1 = aff.sum(np.triu(A_cl))
        assert Expression.are_equivalent(expr0, expr1.value)