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


class TestToys1(unittest.TestCase):

    def _geometric_program_1(self, solver, **kwargs):
        """
        Solve a GP with a linear objective and single posynomial constraint.

        The reference solution was computed by Wolfram Alpha.
        """
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
        status, val = prob.solve(solver=solver, **kwargs)
        assert status == 'solved'
        assert abs(val - 10.4075826) < 1e-6
        x_star = x.value
        expect = np.array([-4.93083, -2.73838])
        assert np.allclose(x_star, expect, atol=1e-4)
        return prob

    def test_geometric_program_1a(self):
        _ = self._geometric_program_1('ECOS', verbose=False)

    @unittest.skipUnless(cl.Mosek.is_installed(), 'MOSEK-specific option')
    def test_geometric_program_1b(self):
        force_dual = self._geometric_program_1('MOSEK', dualize=True, cache_apply_data=True)
        force_prim = self._geometric_program_1('MOSEK', dualize=False, cache_apply_data=True)
        self.assertIn('A', force_prim.solver_apply_data['MOSEK'][0])
        self.assertNotIn('G', force_prim.solver_apply_data['MOSEK'][0])
        self.assertIn('G', force_dual.solver_apply_data['MOSEK'][0])
        self.assertNotIn('A', force_dual.solver_apply_data['MOSEK'][0])

    def test_simple_sage_1(self):
        """
        Solve a simple SAGE relaxation for a signomial minimization problem.

        Do this without resorting to "Signomial" objects.
        """
        alpha = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 0],
                          [0, 0.5]])
        gamma = cl.Variable(shape=(), name='gamma')
        c = cl.Expression([0 - gamma, 3, 2, 1, -4, -2])
        expected_val = -1.8333331773244161

        # with presolve
        cl.presolve_trivial_age_cones(True)
        con = cl.PrimalSageCone(c, alpha, None, 'test_con_name')
        obj = gamma
        prob = Problem(cl.MAX, obj, [con])
        status, val = prob.solve(solver='ECOS', verbose=False)
        assert abs(val - expected_val) < 1e-6
        v = con.violation()
        assert v < 1e-6

        # without presolve
        cl.presolve_trivial_age_cones(False)
        con = cl.PrimalSageCone(c, alpha, None, 'test_con_name')
        obj = gamma
        prob = Problem(cl.MAX, obj, [con])
        status, val = prob.solve(solver='ECOS', verbose=False)
        assert abs(val - expected_val) < 1e-6
        v = con.violation()
        assert v < 1e-6

    def test_redundant_components(self):
        # create problems where some (but not all) components of a vector variable
        # participate in the final conic formulation.
        x = cl.Variable(shape=(4,))
        cons = [0 <= x[1:], cl.sum(x[1:]) <= 1]
        objective = x[1] + 0.5 * x[2] + 0.25 * x[3]
        prob = cl.Problem(cl.MAX, objective, cons)
        prob.solve(solver='ECOS', verbose=False)
        assert np.allclose(x.value, np.array([0, 1, 0, 0]))
        pass
