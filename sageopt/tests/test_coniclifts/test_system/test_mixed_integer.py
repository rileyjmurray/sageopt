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
from sageopt import coniclifts as cl


@unittest.skipUnless(cl.Mosek.is_installed(), 'We only support mixed-integer with MOSEK.')
class TestMixedInteger(unittest.TestCase):

    def test_trivial_01LP(self):
        x = cl.Variable()
        obj_expr = x
        cont_cons = [0 <= x, x <= 1.5]
        prob = cl.Problem(cl.MAX, obj_expr, cont_cons,
                          integer_variables=[x])
        prob.solve(solver='MOSEK')
        self.assertAlmostEqual(x.value, 1.0, places=5)
        pass

    def test_simple_MILP(self):
        # Include continuous variables
        x = cl.Variable()
        y = cl.Variable((2,))
        obj_expr = y[0]  # minimize me
        cont_cons = [cl.sum(y) == x, -1.5 <= x, x <= 2.5, 0 <= y[1], y[1] <= 4.7]
        prob = cl.Problem(cl.MIN, obj_expr, cont_cons, integer_variables=[x])
        prob.solve(solver='MOSEK')
        # to push y[0] negative, we need to push x to its lower bounds
        # and y[1] to its upper bound.
        expect_y = np.array([-5.7, 4.7])
        expect_x = -1.0
        self.assertAlmostEqual(y[0].value, expect_y[0], places=5)
        self.assertAlmostEqual(y[1].value, expect_y[1], places=5)
        self.assertAlmostEqual(x.value, expect_x, places=5)
        pass

    def test_simple_MINLP(self):
        x = cl.Variable(shape=(3,))
        y = cl.Variable(shape=(2,))
        constraints = [cl.vector2norm(x) <= y[0],
                       cl.vector2norm(x) <= y[1],
                       x[0] + x[1] + 3 * x[2] >= 0.1,
                       y <= 5]
        obj_expr = 3 * x[0] + 2 * x[1] + x[2] + y[0] + 2 * y[1]
        prob = cl.Problem(cl.MIN, obj_expr, constraints, integer_variables=[y])
        prob.solve(solver='MOSEK')

        expect_obj = 0.21363997604807272
        self.assertAlmostEqual(prob.value, expect_obj, places=4)
        expect_x = np.array([-0.78510265, -0.43565177, 0.44025147])
        for i in [0, 1, 2]:
            self.assertAlmostEqual(x[i].value, expect_x[i], places=4)
        expect_y = np.array([1.0, 1.0])
        for i in [0, 1]:
            self.assertAlmostEqual(y[i].value, expect_y[i], places=4)
        pass
