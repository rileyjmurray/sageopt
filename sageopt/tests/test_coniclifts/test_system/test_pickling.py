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
from sageopt import coniclifts as cl
from sageopt.coniclifts.problems.problem import Problem
import pickle


class TestPickling(unittest.TestCase):

    @staticmethod
    def case_1():
        alpha = np.array([[1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 0],
                          [0, 0.5]])
        c = np.array([3, 2, 1, 4, 2])
        x = cl.Variable(shape=(2,), name='x')
        y = alpha @ x
        expr = cl.weighted_sum_exp(c, y)
        cons = [expr <= 1]
        obj = - x[0] - 2 * x[1]
        prob = Problem(cl.MIN, obj, cons)
        status = 'solved'
        value = 10.4075826  # up to 1e-6
        x_star = np.array([-4.93083, -2.73838])  # up to 1e-4
        return prob, status, value, x_star

    def test_case_1_unpickle_then_solve(self):
        prob, expect_status, expect_value, expect_x = self.case_1()
        pickled_prob = pickle.dumps(prob)
        del prob
        cl.clear_variable_indices()
        prob = pickle.loads(pickled_prob)
        res = prob.solve(solver='ECOS', verbose=False)
        assert res[0] == expect_status
        assert abs(res[1] - expect_value) < 1e-6
        x = None
        for v in prob.all_variables:
            if v.name == 'x':
                x = v
                assert x.is_proper()
                break
        x_star = x.value
        assert np.allclose(x_star, expect_x, atol=1e-4)

    def test_case_1_solve_then_pickle_unpickle(self):
        prob, expect_status, expect_value, expect_x = self.case_1()
        prob.solve(solver='ECOS', verbose=False)  # ignore return value
        pickled_prob = pickle.dumps(prob)
        del prob
        prob = pickle.loads(pickled_prob)
        assert prob.status == expect_status
        assert abs(prob.value - expect_value) < 1e-6
        x = None
        for v in prob.all_variables:
            if v.name == 'x':
                assert v.is_proper()
                x = v
                break
        x_star = x.value
        assert np.allclose(x_star, expect_x, atol=1e-4)
