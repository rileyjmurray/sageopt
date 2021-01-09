import numpy as np
import scipy.sparse as sp
from sageopt.coniclifts import utilities as util
from sageopt.coniclifts.cones import build_cone_type_selectors
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.problems.solvers.solver import Solver
import copy

class Cvxpy(Solver):

    @staticmethod
    def apply(c, A, b, K, params):
        """
        """
        data = { 'c': c, 'A': A, 'b': b, 'K': K}
        inv_data = dict()
        return data, inv_data

    @staticmethod
    def solve_via_data(data, params):
        import cvxpy as cp
        # linear program solving with cvxpy
        c = data['c']
        b = data['b']
        A = data['A']
        
        m, n = A.shape
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T@x), [A @ x + b >= 0])
        prob.solve()
        return. prob

    # noinspection SpellCheckingInspection
    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        """
        :param inv_data: not used for Cvxpy.
        :param solver_output: the return value for 

        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: problem_status, variable_values, problem_value. The first of
        these is a string (coniclifts.solved, coniclifts.inaccurate, or coniclifts.failed).
        The second of these is a dictionary from coniclifts Variable names to numpy arrays
        containing the values of those Variables at the returned solution. The last of
        these is either a real number, or +/-np.Inf, or np.NaN.
        """
        prob = solver_output
        variable_values = dict()
        if prob.status in ["infeasible", "unbounded"]:
            if prob.status == "infeasible":
                # primal optimal
                problem_status = CL_CONSTANTS.solved
                problem_value = prob.value
            elif prob.status == "unbounded":
                # primal infeasible
                problem_status = CL_CONSTANTS.solved
                problem_value = np.Inf
            else:  # pragma: no cover
                # dual likely infeasible (primal likely unbounded)
                problem_status = CL_CONSTANTS.inaccurate
                problem_value = -np.Inf
        else:
            problem_status = CL_CONSTANTS.solved
            problem_value = prob.value
            for variable in prob.variables():
                variable_values[variable.name()] = variable.value
        # else:
        #     # solver failed, do not record solution
        #     problem_status = CL_CONSTANTS.failed
        #     variable_values = dict()
        #     problem_value = np.NaN
        return problem_status, variable_values, problem_value

    @staticmethod
    def is_installed():
        try:
            import cvxpy as cp
            return True
        except ImportError:
            return False

    @staticmethod
    def load_variable_values(x, var_mapping):
        """
        :param x: the primal solution returned by Cvxpy
        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: a dictionary mapping names of coniclifts Variables to arrays of their
        corresponding values.
        """
        x = np.hstack([x, 0])
        # The final coordinate is a dummy value, which is loaded into ScalarVariables
        # which (1) did not participate in the problem, but (2) whose parent Variable
        # object did participate in the problem.
        var_values = dict()
        for var_name in var_mapping:
            var_values[var_name] = x[var_mapping[var_name]]
        return var_values
