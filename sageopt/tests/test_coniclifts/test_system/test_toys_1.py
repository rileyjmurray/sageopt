import unittest
import numpy as np
from sageopt import coniclifts as cl
from sageopt.coniclifts.problems.problem import Problem


class TestToys1(unittest.TestCase):

    def test_geometric_program_1(self):
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
        solver = 'ECOS'
        res = prob.solve(solver=solver, verbose=False)
        assert res[0] == 'solved'
        assert abs(res[1] - 10.4075826) < 1e-6
        x_star = x.value()
        expect = np.array([-4.93083, -2.73838])
        assert np.allclose(x_star, expect, atol=1e-4)

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
        c = np.array([0 - gamma, 3, 2, 1, -4, -2])
        con = cl.PrimalSageCone(c, alpha, name='words')
        obj = gamma
        prob = Problem(cl.MAX, obj, [con])
        status, val = prob.solve(solver='ECOS', verbose=False)
        expected_val = -1.8333331773244161
        assert abs(val - expected_val) < 1e-6
        v = con.violation()
        assert v < 1e-6


if __name__ == '__main__':
    unittest.main()
