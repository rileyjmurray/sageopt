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
import numpy as np
import scipy.sparse as sp
from sageopt.coniclifts import utilities as util
from sageopt.coniclifts.cones import build_cone_type_selectors
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.problems.solvers.solver import Solver
import copy


class ECOS(Solver):

    @staticmethod
    def apply(c, A, b, K, params):
        """
        :return: G, h, cones, A_ecos, b_ecos --- where we expect
         a function call: sol = ecos.solve(c, G, h, cones, A_ecos, b_ecos)
        """
        for co in K:
            if co.type not in {'e', 'S', '+', '0'}:  # pragma: no cover
                msg1 = 'ECOS only supports cones with labels in the set {"e", "S", "+", "0"}. \n'
                msg2 = 'The provided data includes an invalid cone labeled "' + str(co[0]) + '".'
                raise RuntimeError(msg1 + msg2)
        type_selectors = build_cone_type_selectors(K)
        # find indices of A corresponding to equality constraints
        A_ecos = A[type_selectors['0'], :]
        b_ecos = -b[type_selectors['0']]  # move to RHS
        # Build "G, h".
        G_nonneg = -A[type_selectors['+'], :]
        h_nonneg = b[type_selectors['+']]
        G_soc = -A[type_selectors['S'], :]
        h_soc = b[type_selectors['S']]
        G_exp = -A[type_selectors['e'], :]
        h_exp = b[type_selectors['e']]
        G = sp.vstack([G_nonneg, G_soc, G_exp], format='csc')
        h = np.hstack((h_nonneg, h_soc, h_exp))
        # create cone dims dict for ECOS
        cones = {'l': int(np.sum(type_selectors['+'])),
                 'e': int(np.sum(type_selectors['e']) / 3),
                 'q': util.contiguous_selector_lengths(type_selectors['S'])}
        data = {'G': G, 'h': h, 'cones': cones, 'A': A_ecos, 'b': b_ecos, 'c': c}
        inv_data = dict()
        return data, inv_data

    @staticmethod
    def solve_via_data(data, params):
        import ecos
        if 'max_iters' in params:
            max_iters = params['max_iters']
        else:
            max_iters = 100000
        sol = ecos.solve(data['c'], data['G'], data['h'], data['cones'], data['A'], data['b'],
                         verbose=params['verbose'],
                         max_iters=max_iters)
        return sol

    # noinspection SpellCheckingInspection
    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        """
        :param inv_data: not used for ECOS.
        :param solver_output: a dictionary returned by the ecos package's "csolve" function.
        The most important fields are "solver_output['x']", which contains the primal optimal
        variable (when it exists), and "solver_output['info']", which contains exit flags
        and solution metrics.

        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: problem_status, variable_values, problem_value. The first of
        these is a string (coniclifts.solved, coniclifts.inaccurate, or coniclifts.failed).
        The second of these is a dictionary from coniclifts Variable names to numpy arrays
        containing the values of those Variables at the returned solution. The last of
        these is either a real number, or +/-np.Inf, or np.NaN.

        Notes: solver_output is described in
            https://github.com/embotech/ecos-python#calling-ecos-from-python
        and solver_output['info'] is described in
            https://www.embotech.com/ECOS/Matlab-Interface/Matlab-Native#MATLAB_API_SolveStatus
        """
        solver_status = solver_output['info']['exitFlag']
        variable_values = dict()
        if solver_status in {0, 1, 2, 10, 11, 12}:
            if solver_status == 0:
                # primal optimal
                problem_status = CL_CONSTANTS.solved
                problem_value = solver_output['info']['pcost']
                variable_values = ECOS.load_variable_values(solver_output['x'], var_mapping)
            elif solver_status == 1:
                # primal infeasible
                problem_status = CL_CONSTANTS.solved
                problem_value = np.Inf
            elif solver_status == 2:
                # dual infeasible (primal unbounded)
                problem_status = CL_CONSTANTS.solved
                problem_value = -np.Inf
            elif solver_status == 10:  # pragma: no cover
                # primal near-optimal
                problem_status = CL_CONSTANTS.inaccurate
                problem_value = solver_output['info']['pcost']
                variable_values = ECOS.load_variable_values(solver_output['x'], var_mapping)
            elif solver_status == 11:  # pragma: no cover
                # primal likely infeasible
                problem_status = CL_CONSTANTS.inaccurate
                problem_value = np.Inf
            else:  # pragma: no cover
                # dual likely infeasible (primal likely unbounded)
                problem_status = CL_CONSTANTS.inaccurate
                problem_value = -np.Inf
        else:
            # solver failed, do not record solution
            #    -1	Maximum number of iterations reached	ECOS_MAXIT
            #    -2	Numerical problems (unreliable search direction)	ECOS_NUMERICS
            #    -3	Numerical problems (slacks or multipliers outside cone)	ECOS_OUTCONE
            #    -4	Interrupted by signal or CTRL-C	ECOS_SIGINT
            #    -7	Unknown problem in solver	ECOS_FATAL
            problem_status = CL_CONSTANTS.failed
            variable_values = dict()
            problem_value = np.NaN
        return problem_status, variable_values, problem_value

    @staticmethod
    def is_installed():
        try:
            import ecos
            return True
        except ImportError:
            return False

    @staticmethod
    def load_variable_values(x, var_mapping):
        """
        :param x: the primal solution returned by ECOS.
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
