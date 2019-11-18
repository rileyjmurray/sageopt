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
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.operators.exp import weighted_sum_exp
from sageopt.coniclifts import MIN as CL_MIN
from sageopt.coniclifts.reformulators import separate_cone_constraints


class TestReformulators(unittest.TestCase):

    def test_separate_cone_constraints_1(self):
        num_ineqs = 10
        num_vars = 5
        G = np.random.randn(num_ineqs, num_vars).round(decimals=3)
        x = Variable(shape=(num_vars,))
        h = np.random.randn(num_ineqs).round(decimals=3)
        cons = [G @ x >= h]
        prob = Problem(CL_MIN, Expression([0]), cons)
        A0, b0, K0 = prob.A, prob.b, prob.K
        # main test (separate everything other than the zero cone)
        A1, b1, K1, sepK1 = separate_cone_constraints(A0, b0, K0)
        A1 = A1.toarray()
        assert A1.shape == (num_ineqs, num_vars + num_ineqs)
        expect_A1 = np.hstack((G, -np.eye(num_ineqs)))
        assert np.allclose(A1, expect_A1)
        assert len(K1) == 1
        assert K1[0].type == '0'
        assert len(sepK1) == 1
        assert sepK1[0].type == '+'
        assert np.allclose(b0, b1)
        assert np.allclose(b0, -h)
        # trivial test (don't separate anything, including some cones not in the set)
        A2, b2, K2, sepK2 = separate_cone_constraints(A0, b0, K0, dont_sep={'+', '0', 'S', 'e'})
        A2 = A2.toarray()
        A0 = A0.toarray()
        assert np.allclose(A0, A2)
        assert np.allclose(b0, b2)
        pass

    def test_separate_cone_constraints_2(self):
        num_vars = 5
        x = Variable(shape=(num_vars,))
        cons = [vector2norm(x) <= 1]
        prob = Problem(CL_MIN, Expression([0]), cons)
        A0, b0, K0 = prob.A, prob.b, prob.K
        assert A0.shape == (num_vars + 2, num_vars + 1)
        assert len(K0) == 2
        assert K0[0].type == '+' and K0[0].len == 1
        assert K0[1].type == 'S' and K0[1].len == num_vars + 1
        A1, b1, K1, sepK1 = separate_cone_constraints(A0, b0, K0, dont_sep={'+'})
        A1 = A1.toarray()
        assert A1.shape == (num_vars + 2, 2*(num_vars + 1))
        assert len(K1) == 2
        assert K1[0].type == '+' and K1[0].len == 1
        assert K1[1].type == '0' and K1[1].len == num_vars + 1
        assert len(sepK1) == 1
        assert sepK1[0].type == 'S' and sepK1[0].len == num_vars + 1
        A0 = A0.toarray()
        temp = np.vstack((np.zeros(shape=(1, num_vars + 1)), np.eye(num_vars + 1)))
        expect_A1 = np.hstack((A0, -temp))
        assert np.allclose(expect_A1, A1)

    def test_separate_cone_constraints_3(self):
        alpha = np.array([[1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 0],
                          [0, 0.5]])
        c = np.array([3, 2, 1, 4, 2])
        x = Variable(shape=(2,), name='x')
        expr = weighted_sum_exp(c, alpha @ x)
        cons = [expr <= 1]
        obj = - x[0] - 2 * x[1]
        prob = Problem(CL_MIN, obj, cons)
        A0, b0, K0 = prob.A, prob.b, prob.K
        assert A0.shape == (16, 7)
        assert len(K0) == 6
        assert K0[0].type == '+' and K0[0].len == 1
        for i in [1, 2, 3, 4, 5]:
            assert K0[i].type == 'e'
        A1, b1, K1, sepK1 = separate_cone_constraints(A0, b0, K0, dont_sep={'+'})
        A1 = A1.toarray()
        A0 = A0.toarray()
        temp = np.vstack((np.zeros(shape=(1, 15)), np.eye(15)))
        expect_A1 = np.hstack((A0, -temp))
        assert np.allclose(A1, expect_A1)
        assert len(K1) == len(K0)
        assert K1[0].type == '+' and K1[0].len == 1
        for i in [1, 2, 3, 4, 5]:
            assert K1[i].type == '0' and K1[i].len == 3
        assert len(sepK1) == 5
        for co in sepK1:
            assert co.type == 'e'
        pass


