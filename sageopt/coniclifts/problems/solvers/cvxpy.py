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
        data = {'c': c, 'A': A, 'b': b, 'K': K}
        inv_data = dict()
        return data, inv_data

    @staticmethod
    def solve_via_data(data, params):
        import cvxpy as cp
        # linear program solving with cvxpy
        c = data['c']
        b = data['b']
        A = data['A']
        K = data['K']
        m, n = A.shape
        x = cp.Variable(n)
        for co in K:
            if co.type not in {'e', 'S', '+', '0', 'pow'}:  # pragma: no cover
                msg = """
                The CVXPY interface only supports cones with labels in the set
                    {"e", "S", "+", "0", "pow"}.
                The provided data includes an invalid cone labeled %s.
                """ % str(co[0])
                raise RuntimeError(msg)
        type_selectors = build_cone_type_selectors(K)
        constraints = []
        cone_types = type_selectors.keys()
        if '0' in cone_types:
            tss = type_selectors['0']
            constraints.append(A[tss, :] @ x + b[tss] == 0)
        if 'e' in cone_types:
            rows = np.where(type_selectors['e'])[0][::3]
            expr1 = A[rows, :] @ x + b[rows]
            rows += 1
            expr2 = A[rows, :] @ x + b[rows]
            rows += 1
            expr3 = A[rows, :] @ x + b[rows]
            constraints.append(cp.constraints.ExpCone(expr1, expr3, expr2))
        if '+' in cone_types:
            tss = type_selectors['+']
            constraints.append(A[tss, :] @ x + b[tss] >= 0)
        if 'S' in cone_types or 'pow' in cone_types:
            idx = 0
            for co in K:
                if co.type == 'S':
                    # first component is epigraph variable
                    upper = A[idx, :] @ x + b[idx]
                    start = idx + 1
                    lower = A[start:(idx + co.len), :] @ x + b[start:(idx + co.len)]
                    # upper >= norm(lower, 2)
                    constraints.append(cp.constraints.SOC(upper, lower))
                elif co.type == 'pow':
                    # final component is hypograph variable
                    stop = idx + (co.len - 1)
                    upper = A[idx:stop, :] @ x + b[idx:stop]
                    lower = A[stop, :] @ x + b[stop]  # inclusive
                    weights = co.annotations['weights']
                    # np.prod(np.power(upper, weights)) >= abs(lower); upper >= 0.
                    constraints.append(cp.constraints.PowConeND(upper, lower, weights))
                idx += co.len
        prob = cp.Problem(cp.Minimize(c @ x), constraints)
        prob.solve()
        return x, prob

    # noinspection SpellCheckingInspection
    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        """
        :param inv_data: not used for Cvxpy.
        :param solver_output: the return value for solve_via_data
            unpacked into two values, x_var and param

        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: problem_status, variable_values, problem_value. The first of
        these is a string (coniclifts.solved, coniclifts.inaccurate, or coniclifts.failed).
        The second of these is a dictionary from coniclifts Variable names to numpy arrays
        containing the values of those Variables at the returned solution. The last of
        these is either a real number, or +/-np.Inf, or np.NaN.
        """
        import cvxpy as cp
        x_var, prob = solver_output
        # x_var is in the form of a cvxpy Variable, needs to be converted to np array
        x = x_var.value
        variable_values = dict()
        problem_value = prob.value
        if prob.status in cp.settings.SOLUTION_PRESENT:
            problem_status = CL_CONSTANTS.solved
            variable_values = Cvxpy.load_variable_values(x, var_mapping)
        elif prob.status in cp.settings.INF_OR_UNB:
            problem_status = CL_CONSTANTS.solved
        else:
            problem_status = CL_CONSTANTS.failed
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
