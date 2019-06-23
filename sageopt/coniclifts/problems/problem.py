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
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.problems.solvers.ecos import ECOS
from sageopt.coniclifts.problems.solvers.mosek import Mosek
from sageopt.coniclifts.compilers import compile_problem
from sageopt.coniclifts.base import Expression
import numpy as np
import time


class Problem(object):

    _SOLVERS_ = {'ECOS': ECOS(), 'MOSEK': Mosek()}

    _SOLVER_ORDER_ = ['MOSEK', 'ECOS']

    def __init__(self, objective_sense, objective_expression, constraints):
        """
        :param objective_sense: either coniclifts.minimize or coniclifts.maximize.
        :param objective_expression: A real numeric type, or Expression with .size == 1, or ScalarExpression.
        :param constraints: a list of coniclifts Constraint objects.
        """
        self.objective_sense = objective_sense
        if not isinstance(objective_expression, Expression):
            objective_expression = Expression(objective_expression)
        self.user_obj = objective_expression
        self.user_cons = constraints
        self.timings = dict()
        t = time.time()
        c, A, b, K, sep_K, var_name_to_locs, all_vars = compile_problem(objective_expression, constraints)
        if objective_sense == CL_CONSTANTS.minimize:
            self.c = c
        else:
            self.c = -c
        self.timings['compile_time'] = time.time() - t
        self.A = A
        self.b = b
        self.K = K
        self.sep_K = sep_K
        self.all_variables = all_vars
        self.user_variable_map = var_name_to_locs
        self.user_variable_values = None
        self.solver_apply_data = dict()
        self.solver_runtime_data = dict()
        self.status = None  # "solved", "inaccurate", or "failed"
        self.value = np.NaN
        self.associated_data = dict()
        self.default_options = {'cache_applytime': False, 'cache_solvetime': False,
                                'destructive': False, 'compilation_options': dict(),
                                'verbose': True}
        pass

    def solve(self, solver=None, **kwargs):
        """
        :param solver: a string. 'MOSEK' or 'ECOS'.
        :return:
        """
        if solver is None:
            for svr in Problem._SOLVER_ORDER_:
                if Problem._SOLVERS_[svr].is_installed():
                    solver = svr
                    break
        if solver is None:
            raise RuntimeError('No acceptable solver is installed.')
        options = self.default_options.copy()
        options.update(kwargs)
        solver_object = Problem._SOLVERS_[solver]
        if not solver_object.is_installed():
            raise RuntimeError('Solver "' + solver + '" is not installed.')
        self.timings[solver] = dict()

        # Finish solver-specific compilation
        t0 = time.time()
        data, inv_data = solver_object.apply(self.c, self.A, self.b, self.K, self.sep_K,
                                             options['destructive'],
                                             options['compilation_options'])
        self.timings[solver]['apply'] = time.time() - t0
        if options['cache_applytime']:
            self.solver_apply_data[solver] = data

        # Solve the problem
        t1 = time.time()
        raw_result = solver_object.solve_via_data(data, options)
        self.timings[solver]['solve_via_data'] = time.time() - t1
        if options['cache_solvetime']:
            self.solver_runtime_data[solver] = raw_result
        parsed_result = solver_object.parse_result(raw_result, inv_data, self.user_variable_map)
        self.status = parsed_result[0]
        self.user_variable_values = parsed_result[1]

        # Load values into ScalarVariable objects.
        if len(self.user_variable_values) > 0:
            for v in self.all_variables:
                if v.name in self.user_variable_values:
                    var_val = self.user_variable_values[v.name]
                    v.set_scalar_variables(var_val)

        # Record objective value
        if self.status in {CL_CONSTANTS.solved, CL_CONSTANTS.inaccurate}:
            if self.objective_sense == CL_CONSTANTS.minimize:
                self.value = parsed_result[2]
            else:
                self.value = -parsed_result[2]
        else:
            self.value = np.NaN

        self.timings[solver]['total'] = time.time() - t0
        return self.status, self.value





