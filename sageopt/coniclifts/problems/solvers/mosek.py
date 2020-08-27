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
from sageopt.coniclifts.reformulators import separate_cone_constraints, dualize_problem
from sageopt.coniclifts.problems.solvers.solver import Solver
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.cones import Cone, build_cone_type_selectors


class Mosek(Solver):
    # TODO: fix the docstrings for functions in this class.

    _SDP_SUPPORT_ = False

    @staticmethod
    def decide_primal_vs_dual(c, A, b, K, params):
        if 'integers' in params:
            return 'primal'
        if 'dualize' in params:
            if params['dualize']:
                return 'dual'
            else:
                return 'primal'
        slack_dim = sum([Ki.len for Ki in K if Ki.type in {'e', 'S'}])
        if slack_dim > A.shape[1]:
            return 'dual'
        else:
            return 'primal'

    @staticmethod
    def apply(c, A, b, K, params):
        # This function is analogous to "apply(...)" in cvxpy's mosek_conif.py.
        """
        # Main obstacle: even after running (A,b,K) through "separate_cone_constraints", the PSD constraints are
        # still problematic. Reason being, MOSEK doesn't work with vectorized semidefinite variables. It assumes
        # that these are declared separately from the vectorized variable, and the contribution of semidefinite
        # variables to linear [in]equalities must be specified separately from the contributions of the vectorized
        # variable.
        #
        # Plan: call "separate_cone_constraints(...)" with dont_sep={'0','+','P'}, and then handle the semidefinite
        # constraints just like I do in CVXPY. For CVXPY I sort the cone types in (A,b,K) so that linear matrix
        # inequalities come last. Then for every LMI I declare a slack variable.
        # I would then essentially copy the flow of
        # https://github.com/cvxgrp/cvxpy/blob/16ad9adad944d4cb34275e2f7dd0e27788b47b18/ ...
        #       .... cvxpy/reductions/solvers/conic_solvers/mosek_conif.py#L389
        # in order to enforce proper equality constraints on these slack variables.
        """
        if not Mosek._SDP_SUPPORT_:
            if any([co.type == 'P' for co in K]):  # pragma: no cover
                raise NotImplementedError('This functionality is being put on hold.')
        form = Mosek.decide_primal_vs_dual(c, A, b, K, params)
        if form == 'primal':
            data, inv_data = Mosek._primal_apply(c, A, b, K)
        else:
            data, inv_data = Mosek._dual_apply(c, A, b, K)
        data['form'] = form
        inv_data['form'] = form
        return data, inv_data

    @staticmethod
    def solve_via_data(data, params):
        # This function is analogous to "solve_via_data(...)" in cvxpy's mosek_conif.py.
        if data['form'] == 'primal':
            solver_output = Mosek._primal_solve_via_data(data, params)
        else:
            solver_output = Mosek._dual_solve_via_data(data, params)
        return solver_output

    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        """
        :param solver_output: a dictionary containing the mosek Task and Environment that
        were constructed at solve-time. Also contains any parameters that were passed at solved-time.

        :param inv_data: a dictionary with key 'n'.

        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: problem_status, variable_values, problem_value. The first of these is a string
        (coniclifts.solved, coniclifts.inaccurate, or coniclifts.failed). The second is a dictionary
        from coniclifts Variable names to numpy arrays containing the values of those Variables at
        the returned solution. The last of these is either a real number, or +/-np.Inf, or np.NaN.
        """
        if inv_data['form'] == 'primal':
            ps, vv, pv = Mosek._primal_parse_result(solver_output, inv_data, var_mapping)
        else:
            ps, vv, pv = Mosek._dual_parse_result(solver_output, inv_data, var_mapping)
        return ps, vv, pv

    @staticmethod
    def _primal_apply(c, A, b, K):
        inv_data = {'n': A.shape[1]}
        A, b, K, sep_K = separate_cone_constraints(A, b, K, dont_sep={'0', '+'})
        c = np.hstack([c, np.zeros(shape=(A.shape[1] - len(c)))])
        type_selectors = build_cone_type_selectors(K)
        # Below: inequality constraints that the "user" intended to give to MOSEK.
        A_ineq = A[type_selectors['+'], :]
        b_ineq = b[type_selectors['+']]
        # Below: equality constraints that the "user" intended to give to MOSEK
        A_z = A[type_selectors['0'], :]
        b_z = b[type_selectors['0']]
        # Finally: the matrix "A" and vector "u_c" that appear in the MOSEK documentation as the standard form for
        # an SDP that includes vectorized variables. We use "b" instead of "u_c". It's value in the MOSEK Task will
        # later be specified with MOSEK's "putconboundlist" function.
        A = -sp.vstack([A_ineq, A_z], format='csc')
        b = np.hstack([b_ineq, b_z])
        K = [Cone('+', A_ineq.shape[0]), Cone('0', A_z.shape[0])]
        # Return values
        data = {'A': A, 'b': b, 'K': K, 'sep_K': sep_K, 'c': c}
        return data, inv_data

    @staticmethod
    def _primal_solve_via_data(data, params):
        import mosek
        env = mosek.Env()
        task = env.Task(0, 0)
        if params['verbose']:
            Mosek.set_verbosity_params(env, task)
        if 'mosek_params' in params:
            Mosek.set_task_params(task, params['mosek_params'])

        # The following lines recover problem parameters, and define helper constants.
        c, A, b, K, sep_K = data['c'], data['A'], data['b'], data['K'], data['sep_K']
        m, n = A.shape

        # Define variables and cone constraints
        task.appendvars(n)
        task.putvarboundlist(np.arange(n, dtype=int),
                             [mosek.boundkey.fr] * n, np.zeros(n), np.zeros(n))
        if 'integer_indices' in data:
            int_idxs = data['integer_indices']
            vartypes = [mosek.variabletype.type_int] * len(int_idxs)
            task.putvartypelist(int_idxs, vartypes)
        for co in sep_K:
            # TODO: vectorize this, by using task.appendcones()
            if co.type == 'S':
                indices = co.annotations['col mapping']
                task.appendcone(mosek.conetype.quad, 0.0, indices)
            elif co.type == 'e':
                co_cols = co.annotations['col mapping']
                indices = [co_cols[1], co_cols[2], co_cols[0]]
                task.appendcone(mosek.conetype.pexp, 0.0, indices)
            else:
                raise RuntimeError('Unknown separated cone ' + str(co[0]) + '.')

        # Define linear inequality and equality constraints.
        task.appendcons(m)
        rows, cols, vals = sp.find(A)
        task.putaijlist(rows.tolist(), cols.tolist(), vals.tolist())
        type_constraint = [mosek.boundkey.up] * K[0].len + [mosek.boundkey.fx] * K[1].len
        task.putconboundlist(np.arange(m, dtype=int), type_constraint, b, b)

        # Define the objective
        task.putclist(np.arange(len(c)), c)
        task.putobjsense(mosek.objsense.minimize)

        # Optimize the Mosek Task and return the result.
        task.optimize()
        if params['verbose']:
            task.solutionsummary(mosek.streamtype.msg)

        solver_output = {'env': env, 'task': task, 'params': params,
                         'integer': 'integer_indices' in data}

        return solver_output

    @staticmethod
    def _primal_parse_result(solver_output, inv_data, var_mapping):
        import mosek
        task = solver_output['task']
        if solver_output['integer']:
            sol = mosek.soltype.itg
        else:
            sol = mosek.soltype.itr
        solution_status = task.getsolsta(sol)
        variable_values = dict()
        if solution_status in [mosek.solsta.optimal, mosek.solsta.integer_optimal]:
            # optimal
            problem_status = CL_CONSTANTS.solved
            problem_value = task.getprimalobj(sol)
            x0 = [0.] * inv_data['n']
            task.getxxslice(sol, 0, len(x0), x0)
            x0 = np.array(x0)
            variable_values = Mosek.load_variable_values(x0, inv_data, var_mapping)
        elif solution_status == mosek.solsta.dual_infeas_cer:
            # unbounded
            problem_status = CL_CONSTANTS.solved
            problem_value = -np.Inf
        elif solution_status == mosek.solsta.prim_infeas_cer:
            # infeasible
            problem_status = CL_CONSTANTS.solved
            problem_value = np.Inf
        else:  # pragma: no cover
            # some kind of solver failure.
            problem_status = CL_CONSTANTS.failed
            variable_values = dict()
            problem_value = np.NaN
        return problem_status, variable_values, problem_value

    @staticmethod
    def _dual_apply(c, A, b, K):
        f, G, h, Kd = dualize_problem(c, A, b, K)  # max{ f @ y : G @ y == h, y in Kd}
        type_selectors = build_cone_type_selectors(Kd)
        G_ineq = G[:, type_selectors['+']]
        f_ineq = f[type_selectors['+']]
        G_free = G[:, type_selectors['fr']]
        f_free = f[type_selectors['fr']]
        G_dexp = G[:, type_selectors['de']]
        f_dexp = f[type_selectors['de']]
        G_soc = G[:, type_selectors['S']]
        f_soc = f[type_selectors['S']]
        f = np.concatenate((f_ineq, f_soc, f_dexp, f_free))
        G = sp.hstack((G_ineq, G_soc, G_dexp, G_free), format='csc')
        cones = {'+': np.count_nonzero(type_selectors['+']),
                 'S': [Ki.len for Ki in Kd if Ki.type == 'S'],
                 'de': len([Ki for Ki in Kd if Ki.type == 'de']),
                 'fr': np.count_nonzero(type_selectors['fr'])}
        inv_data = {'A': A, 'b': b, 'K': K, 'c': c,
                    'type_selectors': type_selectors, 'dual': True, 'n': A.shape[1]}
        data = {'f': f, 'G': G, 'h': h, 'cone_dims': cones}
        return data, inv_data

    @staticmethod
    def _dual_solve_via_data(data, params):
        import mosek
        env = mosek.Env()
        task = env.Task(0, 0)
        if params['verbose']:
            Mosek.set_verbosity_params(env, task)
        if 'mosek_params' in params:
            Mosek.set_task_params(task, params['mosek_params'])
        # problem data
        f, G, h, cone_dims = data['f'], data['G'], data['h'], data['cone_dims']
        n, m = G.shape
        task.appendvars(m)
        zero = np.zeros(m)
        task.putvarboundlist(np.arange(m, dtype=int), [mosek.boundkey.fr] * m, zero, zero)
        task.appendcons(n)
        # objective
        task.putclist(np.arange(f.size, dtype=int), f)
        task.putobjsense(mosek.objsense.maximize)
        # equality constraints
        rows, cols, vals = sp.find(G)
        task.putaijlist(rows.tolist(), cols.tolist(), vals.tolist())
        task.putconboundlist(np.arange(n, dtype=int), [mosek.boundkey.fx] * n, h, h)
        # conic constraints
        idx = 0
        m_pos = cone_dims['+']
        if m_pos > 0:
            zero = np.zeros(m_pos)
            task.putvarboundlist(np.arange(m_pos, dtype=int), [mosek.boundkey.lo] * m_pos, zero, zero)
            idx += m_pos
        num_soc = len(cone_dims['S'])
        if num_soc > 0:
            task.appendconesseq([mosek.conetype.quad]*num_soc, [0]*num_soc, cone_dims['S'], idx)
            idx += sum(cone_dims['S'])
        num_exp = cone_dims['de']
        if num_exp > 0:
            for i in range(num_exp):
                # in coniclifts standard, apply constraint to [idx, idx+1, idx+2]
                task.appendcone(mosek.conetype.dexp, 0, [idx+1, idx+2, idx])
                idx += 3
        # Optimize the Mosek Task and return the result.
        task.optimize()
        if params['verbose']:
            task.solutionsummary(mosek.streamtype.msg)
        solver_output = {'env': env, 'task': task, 'params': params}
        return solver_output

    @staticmethod
    def _dual_parse_result(solver_output, inv_data, var_mapping):
        import mosek
        task = solver_output['task']
        sol = mosek.soltype.itr
        solution_status = task.getsolsta(sol)
        variable_values = dict()
        if solution_status in [mosek.solsta.optimal, mosek.solsta.integer_optimal]:
            # optimal
            problem_status = CL_CONSTANTS.solved
            problem_value = task.getprimalobj(sol)
            x0 = [0.] * inv_data['n']
            task.gety(sol, x0)
            x0 = np.array(x0)
            variable_values = Mosek.load_variable_values(x0, inv_data, var_mapping)
        elif solution_status == mosek.solsta.dual_infeas_cer:
            # unbounded
            problem_status = CL_CONSTANTS.solved
            problem_value = np.Inf
        elif solution_status == mosek.solsta.prim_infeas_cer:
            # infeasible
            problem_status = CL_CONSTANTS.solved
            problem_value = -np.Inf
        else:  # pragma: no cover
            # some kind of solver failure.
            problem_status = CL_CONSTANTS.failed
            variable_values = dict()
            problem_value = np.NaN
        return problem_status, variable_values, problem_value

    @staticmethod
    def is_installed():
        try:
            import mosek
            return True
        except ImportError:  # pragma: no cover
            return False

    @staticmethod
    def set_verbosity_params(env, task):
        # If verbose, then set default logging parameters.
        import mosek
        import sys

        def stream_printer(text):
            sys.stdout.write(text)
            sys.stdout.flush()
        print('\n')
        env.set_Stream(mosek.streamtype.log, stream_printer)
        task.set_Stream(mosek.streamtype.log, stream_printer)
        task.putintparam(mosek.iparam.log_presolve, 2)

    @staticmethod
    def load_variable_values(x0, inv_data, var_mapping):
        x = np.hstack([x0[:inv_data['n']], 0])
        # The final coordinate is a dummy value, which is loaded into ScalarVariables
        # which (1) did not participate in the problem, but (2) whose parent Variable
        # object did participate in the problem.
        var_values = dict()
        for var_name in var_mapping:
            var_values[var_name] = x[var_mapping[var_name]]
        return var_values

    @staticmethod
    def set_task_params(task, params):
        if params is None:
            return

        import mosek

        params = params.copy()
        task.putintparam(mosek.iparam.num_threads, 0)  # default to all threads.
        if 'NUM_THREADS' in params:
            task.putintparam(mosek.iparam.num_threads, params['NUM_THREADS'])
            del params['NUM_THREADS']
        if 'CO_TOL_NEAR_REL' in params:
            ctnr_param = params['CO_TOL_NEAR_REL']  # multiplicative factor; MOSEK default is 1000
            task.putdouparam(mosek.dparam.intpnt_co_tol_near_rel, ctnr_param)
            del params['CO_TOL_NEAR_REL']
        if 'TOL_PATH' in params:
            tol_path_param = params['TOL_PATH']  # double, in interval (0, 1)
            task.putdouparam(mosek.dparam.intpnt_tol_path, tol_path_param)
            del params['TOL_PATH']
        if 'TOL_STEP_SIZE' in params:
            tol_step_param = params['TOL_STEP_SIZE']  # double, in interval (0, 1)
            task.putdouparam(mosek.dparam.intpnt_tol_step_size, tol_step_param)
            del params['TOL_STEP_SIZE']
        if 'DEACTIVATE_SCALING' in params:
            if params['DEACTIVATE_SCALING'] and isinstance(params['DEACTIVATE_SCALING'], bool):
                task.putintparam(mosek.iparam.intpnt_scaling, mosek.scalingtype.none)
            del params['DEACTIVATE_SCALING']

        def _handle_str_param(param, value):
            if param.startswith("MSK_DPAR_"):
                task.putnadouparam(param, value)
            elif param.startswith("MSK_IPAR_"):
                task.putnaintparam(param, value)
            elif param.startswith("MSK_SPAR_"):
                task.putnastrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        def _handle_enum_param(param, value):
            if isinstance(param, mosek.dparam):
                task.putdouparam(param, value)
            elif isinstance(param, mosek.iparam):
                task.putintparam(param, value)
            elif isinstance(param, mosek.sparam):
                task.putstrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        for param, value in params.items():
            if isinstance(param, str):
                _handle_str_param(param.strip(), value)
            else:
                _handle_enum_param(param, value)
