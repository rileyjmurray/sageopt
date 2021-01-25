import unittest
import numpy as np
import time
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.problems.solvers.ecos import ECOS
from sageopt.coniclifts.problems.solvers.mosek import Mosek
from sageopt.coniclifts.compilers import compile_problem
from sageopt.coniclifts.base import Expression, Variable
from sageopt.coniclifts import Problem, Constraint
import sageopt.coniclifts as cl

class TestProblem(unittest.TestCase):
    def test_solve(self):
        # random problem data
        G = np.random.randn(3, 6)
        h = G @ np.random.rand(6)
        c = np.random.rand(6)
        # input to coniclift's Problem constructor
        x = cl.Variable(shape=(6,))
        constrs = [0 <= x, G @ x == h]
        objective_expression = c @ x
        prob = cl.Problem(cl.MIN, objective_expression, constrs)
        Problem.solve(prob, None)
        # self.assertRaises(RuntimeError, Problem.solve, prob, None)

    def test_parse_integer_constraints(self):
        x = Variable(shape=(3,), name='my_name_x')
        y = Variable(shape=(3,), name='my_name_y')
        z = Variable(shape=(3,), name='my_name_z')
        invalid_lst = [2, 4, 6]
        self.assertRaises(ValueError, Problem._parse_integer_constraints, x, invalid_lst)
        valid_lst = [x, y, z]

        # random problem data
        G = np.random.randn(3, 6)
        h = G @ np.random.rand(6)
        c = np.random.rand(6)
        # input to coniclift's Problem constructor
        x = cl.Variable(shape=(6,))
        constrs = [0 <= x, G @ x == h]
        objective_expression = c @ x
        prob = cl.Problem(cl.MIN, objective_expression, constrs)
        prob.variable_map = {'my_name_x': np.array([0, 1]),
            'my_name_y': np.array([1, 2]),
            'my_name_z': np.array([2, 3])
            }
        prob._parse_integer_constraints(valid_lst)
        int_indices_expected = [0, 1, 1, 2, 2, 3]
        assert Expression.are_equivalent(int_indices_expected, prob._integer_indices)
        prob.variable_map = {'my_name_x': np.array([-1, 1]),
            'my_name_y': np.array([1, 2]),
            'my_name_z': np.array([2, 3])
            }
        self.assertRaises(ValueError, Problem._parse_integer_constraints, prob, valid_lst)

    def test_variables(self):
        # random problem data
        G = np.random.randn(3, 6)
        h = G @ np.random.rand(6)
        c = np.random.rand(6)
        # input to coniclift's Problem constructor
        x = cl.Variable(shape=(6,))
        constrs = [0 <= x, G @ x == h]
        objective_expression = c @ x
        prob = cl.Problem(cl.MIN, objective_expression, constrs)
        x = Variable(shape=(3,), name='my_name')
        shallow_copy = [v for v in prob.all_variables]
        assert Expression.are_equivalent(shallow_copy, prob.variables())