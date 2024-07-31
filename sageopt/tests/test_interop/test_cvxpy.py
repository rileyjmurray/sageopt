"""
   Copyright 2021 Riley John Murray

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
from sageopt.coniclifts.constraints.set_membership.pow_cone import PowCone
from sageopt.coniclifts.problems.problem import Problem
from sageopt.tests.test_coniclifts.helper import SolverTestHelper
from importlib.util import find_spec

CVXPY_INSTALLED = find_spec('cvxpy') is not None


@unittest.skipUnless(CVXPY_INSTALLED, 'These tests are only applicable when CVXPY is installed.')
class TestCVXPY(unittest.TestCase):

    def test_redundant_components(self):
        # create problems where some (but not all) components of a vector variable
        # participate in the final conic formulation.
        x = cl.Variable(shape=(4,))
        cons = [0 <= x[1:], cl.sum(x[1:]) <= 1]
        objective = x[1] + 0.5 * x[2] + 0.25 * x[3]
        prob = cl.Problem(cl.MAX, objective, cons)
        prob.solve(solver='CP', verbose=False)
        assert np.allclose(x.value, np.array([0, 1, 0, 0]))
        pass

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
        _ = self._geometric_program_1('CP', verbose=False)

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
        status, val = prob.solve(solver='CP', verbose=False)
        assert abs(val - expected_val) < 1e-6
        v = con.violation()
        assert v < 1e-6

        # without presolve
        cl.presolve_trivial_age_cones(False)
        con = cl.PrimalSageCone(c, alpha, None, 'test_con_name')
        obj = gamma
        prob = Problem(cl.MAX, obj, [con])
        status, val = prob.solve(solver='CP', verbose=False)
        assert abs(val - expected_val) < 1e-6
        v = con.violation()
        assert v < 1e-6

    def test_pcp_1(self):
        #TODO: reformulate with SolverTestHelper
        """
        Use a 3D power cone formulation for
        min 3 * x[0] + 2 * x[1] + x[2]
        s.t. norm(x,2) <= y
             x[0] + x[1] + 3*x[2] >= 1.0
             y <= 5
        """
        x = cl.Variable(shape=(3,))
        y_square = cl.Variable()
        epis = cl.Variable(shape=(3,))
        constraints = [PowCone(cl.hstack((1.0, x[0], epis[0])), np.array([0.5, -1, 0.5])),
                       PowCone(cl.hstack((1.0, x[1], epis[1])), np.array([0.5, -1, 0.5])),
                       PowCone(cl.hstack((x[2], 1.0, epis[2])), np.array([-1, 0.5, 0.5])),
                        # Could have done PowCone(cl.hstack((1.0, x[2], epis[2])), np.array([0.5, -1, 0.5])).
                       cl.sum(epis) <= y_square,
                       x[0] + x[1] + 3 * x[2] >= 1.0,
                       y_square <= 25]
        objective = 3 * x[0] + 2 * x[1] + x[2]
        expect_x = np.array([-3.874621860638774, -2.129788233677883, 2.33480343377204])
        expect_epis = expect_x ** 2
        expect_x = np.round(expect_x, decimals=5)
        expect_epis = np.round(expect_epis, decimals=5)
        expect_y_square = 25
        expect_objective = -13.548638904065102
        prob = cl.Problem(cl.MIN, objective, constraints)
        prob.solve(solver='CP')
        self.assertAlmostEqual(prob.value, expect_objective, delta=1e-4)
        self.assertAlmostEqual(y_square.value, expect_y_square, delta=1e-4)
        concat = cl.hstack((x.value, epis.value))
        expect_concat = cl.hstack((expect_x, expect_epis))
        for i in range(5):
            self.assertAlmostEqual(concat[i], expect_concat[i], delta=1e-2)
        pass

    def test_pcp_2(self):
        # TODO: reformulate with SolverTestHelper
        """
        Reformulate
            max  (x**0.2)*(y**0.8) + z**0.4 - x
            s.t. x + y + z/2 == 2
                 x, y, z >= 0
        Into
            max  x3 + x4 - x0
            s.t. x0 + x1 + x2 / 2 == 2,
                 (x0, x1, x3) in Pow3D(0.2)
                 (x2, 1.0, x4) in Pow3D(0.4)
        """
        x = cl.Variable(shape=(3,))
        hypos = cl.Variable(shape=(2,))
        objective = -cl.sum(hypos) + x[0]
        con1_expr = cl.hstack((x[0], x[1], hypos[0]))
        con1_weights = np.array([0.2, 0.8, -1.0])
        con2_expr = cl.hstack((x[2], 1.0, hypos[1]))
        con2_weights = np.array([0.4, 0.6, -1.0])
        constraints = [
            x[0] + x[1] + 0.5 * x[2] == 2,
            PowCone(con1_expr, con1_weights),
            PowCone(con2_expr, con2_weights)
        ]
        opt_objective = -1.8073406786220672
        opt_x = np.array([0.06393515, 0.78320961, 2.30571048])
        prob = cl.Problem(cl.MIN, objective, constraints)
        prob.solve(solver='CP')
        self.assertAlmostEqual(prob.value, opt_objective)
        assert np.allclose(x.value, opt_x, atol=1e-3)

    @staticmethod
    def pcp_4(ceei: bool = True):
        """
        A power cone formulation of a Fisher market equilibrium pricing model.
        ceei = Competitive Equilibrium from Equal Incomes
        """
        # Generate test data
        np.random.seed(0)
        n_buyer = 4
        n_items = 6
        V = np.random.rand(n_buyer, n_items)
        X = cl.Variable(shape=(n_buyer, n_items))
        u = cl.sum(V * X, axis=1)
        z = cl.Variable()
        if ceei:
            b = np.ones(n_buyer) / n_buyer
            expect_X = np.array([[9.16311051e-01, 2.71146000e-09, 6.44984275e-10, 0.00000000e+00,
                                1.85098676e-09, 6.66541059e-01],
                               [0.00000000e+00, 0.00000000e+00, 5.30793141e-01, 0.00000000e+00,
                                9.99999995e-01, 1.35828851e-09],
                               [9.78080132e-10, 9.99999998e-01, 0.00000000e+00, 0.00000000e+00,
                                1.16278780e-09, 3.33458937e-01],
                               [8.36889514e-02, 0.00000000e+00, 4.69206858e-01, 1.00000001e+00,
                                7.80694090e-10, 8.26483799e-10]])
            pow_objective = (-z, -1.179743761485325)
        else:
            b = np.array([0.3, 0.15, 0.2, 0.35])
            expect_X = np.array([[9.08798195e-01, 0.00000000e+00, 0.00000000e+00, 2.67738456e-10,
                            3.44073780e-09, 9.58119833e-01],
                           [0.00000000e+00, 1.92431554e-10, 3.91981663e-09, 0.00000000e+00,
                            9.99999991e-01, 0.00000000e+00],
                           [0.00000000e+00, 9.99999993e-01, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 4.18801652e-02],
                           [9.12018094e-02, 1.09687013e-08, 1.00000000e+00, 1.00000001e+00,
                            5.94724468e-09, 6.99603695e-09]])
            pow_objective = (-z, -1.2279371987281384)
        pow_cons = [(cl.sum(X, axis=0) <= 1, None),
                    (PowCone(cl.hstack((u, z)), np.hstack((b, -1))), None),
                    (X >= 0, None)]
        pow_vars = [(X, expect_X)]
        sth = SolverTestHelper(pow_objective, pow_vars, pow_cons)
        return sth

    def test_pcp_4a(self):
        sth = self.pcp_4(ceei=True)
        sth.solve(solver='CP-SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    def test_pcp_4b(self):
        sth = self.pcp_4(ceei=False)
        sth.solve(solver='CP-SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

if __name__ == '__main__':
    import pytest
    pytest.main()
