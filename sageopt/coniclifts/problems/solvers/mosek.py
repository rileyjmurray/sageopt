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
from sageopt.coniclifts.reformulators import separate_cone_constraints, build_cone_type_selectors
from sageopt.coniclifts.problems.solvers.solver import Solver
from sageopt.coniclifts.standards import constants as CL_CONSTANTS
from sageopt.coniclifts.cones import Cone
import copy


class Mosek(Solver):

    _SDP_SUPPORT_ = False

    _CO_TOL_NEAR_REL_ = 1000

    @staticmethod
    def apply(c, A, b, K, sep_K, destructive, compilation_options):
        # This function is analogous to "apply(...)" in cvxpy's mosek_conif.py.
        #
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
        #
        if not Mosek._SDP_SUPPORT_:
            if any([co.type == 'P' for co in K]):
                raise NotImplementedError('This functionality is being put on hold.')
        if not destructive:
            A = A.copy()
            b = b.copy()
            K = copy.deepcopy(K)
            sep_K = copy.deepcopy(sep_K)
            c = c.copy()
        if 'avoid_slacks' in compilation_options:
            avoid_slacks = compilation_options['avoid_slacks']
        else:
            avoid_slacks = False
        A, b, K, sep_K1, scale, trans = separate_cone_constraints(A, b, K,
                                                                  destructive=True,
                                                                  dont_sep={'0', '+', 'P'},
                                                                  avoid_slacks=avoid_slacks)
        sep_K += sep_K1
        # A,b,K now reflect a conic system where "x" has been replaced by "np.diag(scale) @ (x + trans)"
        c = np.hstack([c, np.zeros(shape=(A.shape[1] - len(c)))])
        objective_offset = c @ trans
        c = c * scale
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
        inv_data = {'scaling': scale, 'translation': trans, 'objective_offset': objective_offset, 'n': A.shape[1]}
        return data, inv_data

    @staticmethod
    def solve_via_data(data, params):
        # This function is analogous to "solve_via_data(...)" in cvxpy's mosek_conif.py.
        import mosek
        env = mosek.Env()
        task = env.Task(0, 0)
        if params['verbose']:
            Mosek.set_verbosity_params(env, task)
        Mosek.set_default_solver_settings(task)

        # The following lines recover problem parameters, and define helper constants.
        if not params['destructive']:
            data['A'] = data['A'].copy()
            data['b'] = data['b'].copy()
            data['K'] = data['K'].copy()
        c, A, b, K, sep_K = data['c'], data['A'], data['b'], data['K'], data['sep_K']
        n = A.shape[1]
        m = A.shape[0]

        # Define variables and cone constraints
        task.appendvars(n)
        task.putvarboundlist(np.arange(n, dtype=int),
                             [mosek.boundkey.fr] * n, np.zeros(n), np.zeros(n))
        for co in sep_K:
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
        return {'env': env, 'task': task, 'params': params}

    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        """
        :param solver_output: a dictionary containing the mosek Task and Environment that
        were constructed at solve-time. Also contains any parameters that were passed at solved-time.

        :param inv_data: a dictionary with keys 'scaling' and 'translation'. The primal variable
        values returned by Mosek should be ...
            (1) scaled by the inverse of inv_data['scaling'], and
            (2) translated by the negative of inv_data['translation'].
        Then the objective value should be reduced by inv_data['objective_offset'].

        :param var_mapping: a dictionary mapping names of coniclifts Variables to arrays
        of indices. The array var_mapping['my_var'] contains the indices in
        "solver_output['x']" of the coniclifts Variable object named 'my_var'.

        :return: problem_status, user_variable_values, problem_value. The first of these is a string
        (coniclifts.solved, coniclifts.inaccurate, or coniclifts.failed). The second is a dictionary
        from coniclifts Variable names to numpy arrays containing the values of those Variables at
        the returned solution. The last of these is either a real number, or +/-np.Inf, or np.NaN.
        """
        import mosek
        task = solver_output['task']
        sol = mosek.soltype.itr
        solution_status = task.getsolsta(sol)
        variable_values = dict()
        if solution_status == mosek.solsta.optimal:
            # optimal
            problem_status = CL_CONSTANTS.solved
            problem_value = task.getprimalobj(sol) - inv_data['objective_offset']
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
        else:
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
        except ImportError:
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

    # noinspection SpellCheckingInspection
    @staticmethod
    def set_default_solver_settings(task):
        import mosek
        task.putintparam(mosek.iparam.num_threads, 0)
        task.putdouparam(mosek.dparam.intpnt_co_tol_near_rel, Mosek._CO_TOL_NEAR_REL_)
        # pdsafe = 1e1
        # task.putdouparam(mosek.dparam.intpnt_tol_psafe, pdsafe)
        # task.putdouparam(mosek.dparam.intpnt_tol_dsafe, pdsafe)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, 1e-8)
        # task.putdouparam(mosek.dparam.intpnt_tol_path, 1.0e-1)
        # task.putintparam(mosek.iparam.intpnt_scaling, mosek.scalingtype.free)
        # task.putintparam(mosek.iparam.intpnt_solve_form, mosek.solveform.primal)
        # task.putintparam(mosek.iparam.intpnt_max_num_cor, int(1e8))
        # task.putintparam(mosek.iparam.intpnt_max_num_refinement_steps, int(1e8))
        # task.putintparam(mosek.iparam.intpnt_regularization_use, mosek.onoffkey.on)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_mu_red, 1.0)
        # task.putintparam(mosek.iparam.intpnt_off_col_trh, 5)
        # task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
        # task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, 10)
        # task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.on)
        # task.putintparam(mosek.iparam.presolve_lindep_abs_work_trh, int(1e8))
        # task.putintparam(mosek.iparam.presolve_lindep_rel_work_trh, int(1e8))
        # task.putintparam(mosek.iparam.presolve_eliminator_max_fill, int(1e8))
        # task.putintparam(mosek.iparam.log_response, 3)
        # task.putintparam(mosek.iparam.log_feas_repair, 3)
        pass

    @staticmethod
    def load_variable_values(x0, inv_data, var_mapping):
        x = np.power(inv_data['scaling'], -1) * x0 - inv_data['translation']
        var_values = dict()
        for var_name in var_mapping:
            var_values[var_name] = x[var_mapping[var_name]]
        return var_values
