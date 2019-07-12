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
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.constraints.set_membership.product_cone import PrimalProductCone, DualProductCone
from sageopt.coniclifts.base import Variable, Expression
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.operators import affine as aff
from sageopt.coniclifts.operators.precompiled.relent import sum_relent, elementwise_relent
from sageopt.coniclifts.operators.precompiled import affine as compiled_aff
from sageopt.coniclifts.problems.problem import Problem
import numpy as np
import warnings
from scipy.sparse import issparse
import scipy.special as special_functions

_ALLOWED_CONES_ = {'+', 'S', 'P', 'e', '0'}

_AGGRESSIVE_REDUCTION_ = True


def check_cones(K):
    if any([co.type not in _ALLOWED_CONES_ for co in K]):
        raise NotImplementedError()
    else:
        newK = [Cone(co.type, co.len) for co in K]
        return newK


class PrimalCondSageCone(SetMembership):

    def __init__(self, c, alpha, A, b, K, name, expcovers=None):
        self.name = name
        if issparse(A):
            A = A.toarray()
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.lifted_n = A.shape[1]
        if self.lifted_n > self.n:
            # Then need to zero-pad alpha
            zero_block = np.zeros(shape=(alpha.shape[0], self.lifted_n - self.n))
            alpha = np.hstack((alpha, zero_block))
        self.lifted_alpha = alpha
        self.A = A
        self.b = b
        self.c = Expression(c)  # self.c is now definitely an ndarray of ScalarExpressions.
        self._variables = self.c.variables()
        self.K = check_cones(K)
        if self.m > 1:
            self.ech = ExpCoverHelper(self.lifted_alpha, self.c, self.A, self.b, self.K, expcovers)
            self.nu_vars = dict()
            self.c_vars = dict()
            self.age_vectors = dict()
            self.lambda_vars = dict()
            self._initialize_primary_variables()
        pass

    def _initialize_primary_variables(self):
        for i in self.ech.U_I:
            nu_len = np.count_nonzero(self.ech.expcovers[i])
            if nu_len > 0:
                var_name = 'nu^{(' + str(i) + ')}_' + self.name
                self.nu_vars[i] = Variable(shape=(nu_len,), name=var_name)
            c_len = nu_len
            if i not in self.ech.N_I:
                c_len += 1
            var_name = 'c^{(' + str(i) + ')}_{' + self.name + '}'
            self.c_vars[i] = Variable(shape=(c_len,), name=var_name)
            var_name = 'lambda^{(' + str(i) + ')}_{' + self.name + '}'
            self.lambda_vars[i] = Variable(shape=(self.A.shape[0],), name=var_name)
        self._variables += list(self.nu_vars.values())
        self._variables += list(self.c_vars.values())
        self._variables += list(self.lambda_vars.values())
        pass

    def _build_aligned_age_vectors(self):
        for i in self.ech.U_I:
            ci_expr = Expression(np.zeros(self.m,))
            if i in self.ech.N_I:
                ci_expr[self.ech.expcovers[i]] = self.c_vars[i]
                ci_expr[i] = self.c[i]
            else:
                ci_expr[self.ech.expcovers[i]] = self.c_vars[i][:-1]
                ci_expr[i] = self.c_vars[i][-1]
            self.age_vectors[i] = ci_expr
        pass

    def _age_rel_ent_cone_data(self, i):
        idx_set = self.ech.expcovers[i]
        if np.any(idx_set):
            x = self.nu_vars[i]
            y = np.exp(1) * self.age_vectors[i][idx_set]
            # ^ # This line consumes a disproportionately large amount of runtime
            z = -self.age_vectors[i][i] + self.lambda_vars[i] @ self.b
            name = self.name + '_' + str(i)
            A_vals, A_rows, A_cols, b, K, aux_vars = sum_relent(x, y, z, name)
            if len(aux_vars) > 0:
                self._variables.append(aux_vars)
        else:
            con = self.lambda_vars[i] @ self.b <= self.age_vectors[i][i]
            con.epigraph_checked = True
            A_vals, A_rows, A_cols, b, K, _ = con.conic_form()
        return A_vals, A_rows, A_cols, b, K, []

    def _age_lin_eq_cone_data(self, i):
        idx_set = self.ech.expcovers[i]
        if np.any(idx_set):
            mat1 = (self.lifted_alpha[idx_set, :] - self.lifted_alpha[i, :]).T
            mat2 = -self.A.T
            var1 = self.nu_vars[i]
            var2 = self.lambda_vars[i]
            A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar_plus_mat_times_vecvar(mat1, var1, mat2, var2)
            num_rows = mat1.shape[0] + mat2.shape[0]
        else:
            mat1 = -self.A.T.copy()
            var1 = self.lambda_vars[i]
            A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar(mat1, var1)
            num_rows = mat1.shape[0]
        b = np.zeros(num_rows, )
        K = [Cone('0', num_rows)]
        return A_vals, A_rows, A_cols, b, K, []

    def _age_lambda_var_domain_constraints(self, i):
        con = DualProductCone(self.lambda_vars[i], self.K)
        conic_data = con.conic_form()
        A_vals, A_rows, A_cols, b, cur_K, sep_K = conic_data[0]
        return A_vals, A_rows, A_cols, b, cur_K, sep_K

    def _age_violation(self, i, norm_ord, c_i, lambda_i):
        # This is "rough" only.
        lambda_viol = DualProductCone.project(lambda_i, self.K)
        if np.any(self.ech.expcovers[i]):
            idx_set = self.ech.expcovers[i]
            x_i = self.nu_vars[i].value()
            x_i[x_i < 0] = 0
            y_i = np.exp(1) * c_i[idx_set]
            relent_res = np.sum(special_functions.rel_entr(x_i, y_i)) - c_i[i]  + self.b @ lambda_i # <= 0
            relent_viol = abs(max(relent_res, 0))
            eq_res = (self.lifted_alpha[idx_set, :] - self.lifted_alpha[i, :]).T @ x_i - self.A.T @ lambda_i  # == 0
            eq_res = eq_res.reshape((-1,))
            eq_viol = np.linalg.norm(eq_res, ord=norm_ord)
            total_viol = relent_viol + eq_viol + lambda_viol
            return total_viol
        else:
            c_i = float(self.c_vars[i].value())
            residual = c_i - self.b @ lambda_i  # >= 0
            relent_viol = abs(min(0, residual))
            total_viol = relent_viol + lambda_viol
            return total_viol

    def _age_vectors_sum_to_c(self):
        nonconst_locs = np.ones(self.m, dtype=bool)
        nonconst_locs[self.ech.N_I] = False
        aux_c_vars = list(self.age_vectors.values())
        aux_c_vars = aff.vstack(aux_c_vars).T
        aux_c_vars = aux_c_vars[nonconst_locs, :]
        main_c_var = self.c[nonconst_locs]
        # add constraint that the columns of all_cs sum to self.c[need_constr]
        # TODO: make "sums of vectors equaling another vector" a precompiled atom.
        A_rows, A_cols, A_vals = [], [], []
        num_cons = np.count_nonzero(nonconst_locs)
        b = np.zeros(num_cons,)
        K = [Cone('0', num_cons)]
        for i in range(num_cons):
            # update cols and data to reflect addition of elements in ith row of aux_c_vars
            svs = aux_c_vars[i, :].scalar_variables()
            A_cols += [sv.id for sv in svs]
            A_vals += [1] * len(svs)
            # update cols and data to reflect addition of elements from ith element of main_c_var
            #   ith element of main_c_var is a ScalarExpression!
            id2co = [(a.id, co) for a, co in main_c_var[i].atoms_to_coeffs.items()]
            A_cols += [aid for aid, _ in id2co]
            A_vals += [-co for _, co in id2co]  # we are subtracting, after all.
            # update rows with appropriate number of "i"s.
            A_rows += [i] * (len(svs) + len(id2co))
            # update b
            b[i] -= main_c_var[i].offset
        return A_vals, np.array(A_rows), A_cols, b, K, []

    def variables(self):
        return self._variables

    def conic_form(self):
        if self.m > 1:
            # Lift c_vars and nu_vars into Expressions of length self.m
            self._build_aligned_age_vectors()
            cone_data = []
            # age cones
            for i in self.ech.U_I:
                cone_data.append(self._age_rel_ent_cone_data(i))
                cone_data.append(self._age_lin_eq_cone_data(i))
                cone_data.append(self._age_lambda_var_domain_constraints(i))
            # Vectors sum to s.c
            cone_data.append(self._age_vectors_sum_to_c())
            return cone_data
        else:
            con = self.c >= 0
            con.epigraph_checked = True
            A_vals, A_rows, A_cols, b, K, _ = con.conic_form()
            cone_data = [(A_vals, A_rows, A_cols, b, K, [])]
            return cone_data

    def violation(self, norm_ord=np.inf, rough=False):
        c = self.c.value()
        if self.m > 1:
            if not rough and c in self:
                return 0
            # compute violation for "AGE vectors sum to c"
            #   Although, we can use the fact that the SAGE cone contains R^m_++.
            #   and so only compute violation for "AGE vectors sum to <= c"
            age_vectors = {i: v.value() for i, v in self.age_vectors.items()}
            lambda_vectors = {i: v.value() for i, v in self.lambda_vars.items()}
            sum_age_vectors = sum(age_vectors.values())
            residual = c - sum_age_vectors  # want >= 0
            residual[residual < 0] = 0
            sum_to_c_viol = np.linalg.norm(residual, ord=norm_ord)
            # compute violations for each AGE cone
            age_viols = np.zeros(shape=(len(self.ech.U_I,)))
            for idx, i in enumerate(self.ech.U_I):
                age_viols[idx] = self._age_violation(i, norm_ord, age_vectors[i], lambda_vectors[i])
            # add the max "AGE violation" to the violation for "AGE vectors sum to c".
            total_viol = sum_to_c_viol + np.max(age_viols)
            return total_viol
        else:
            residual = c.reshape((-1,))  # >= 0
            residual[residual >= 0] = 0
            return np.linalg.norm(c, ord=norm_ord)
        pass

    def __contains__(self, item):
        from sageopt.coniclifts import clear_variable_indices, MIN as CL_MIN, SOLVED as CL_SOLVED
        item = Expression(item).ravel()
        name = self.name + ' check membership'
        con = [PrimalCondSageCone(item, self.lifted_alpha, self.A, self.b, self.K, name)]
        prob = Problem(CL_MIN, Expression([0]), con)
        prob.solve(verbose=False)
        clear_variable_indices()
        return prob.status == CL_SOLVED and abs(prob.value) < 1e-7


class DualCondSageCone(SetMembership):

    def __init__(self, v, alpha, A, b, K, name, c=None, expcovers=None):
        """
        Aggregates constraints on "v" so that "v" can be viewed as a dual variable
        to a constraint of the form "c \in C_{SAGE}(alpha, A, b, K)".
        """
        if issparse(A):
            A = A.toarray()
        self.A = A
        self.b = b
        self.K = check_cones(K)
        self.n = alpha.shape[1]
        self.m = alpha.shape[0]
        self.lifted_n = A.shape[1]
        if self.lifted_n > self.n:
            zero_block = np.zeros(shape=(alpha.shape[0], self.lifted_n - self.n))
            alpha = np.hstack((alpha, zero_block))
        self.lifted_alpha = alpha
        self.v = v
        self.name = name
        self._variables = self.v.variables()
        if self.m > 1:
            if c is None:
                self.c = None
            else:
                self.c = Expression(c)
            self.ech = ExpCoverHelper(self.lifted_alpha, self.c, self.A, self.b, self.K, expcovers)
            self.lifted_mu_vars = dict()
            self.mu_vars = dict()
            self._initialize_primary_variables()
        pass

    def _initialize_primary_variables(self):
        for i in self.ech.U_I:
            var_name = 'mu[' + str(i) + ']_{' + self.name + '}'
            self.lifted_mu_vars[i] = Variable(shape=(self.lifted_n,), name=var_name)
            self._variables.append(self.lifted_mu_vars[i])
            self.mu_vars[i] = self.lifted_mu_vars[i][:self.n]
        pass

    def _dual_age_cone_data(self, i):
        cone_data = []
        selector = self.ech.expcovers[i]
        len_sel = np.count_nonzero(selector)
        if len_sel > 0:
            #
            # relative entropy constraints
            #
            expr1 = np.tile(self.v[i], len_sel).view(Expression)
            aux_var_name = 'epi_relent_[' + str(i) + ']_{' + self.name + '}'
            A_vals, A_rows, A_cols, b, K, z = elementwise_relent(expr1, self.v[selector], aux_var_name)
            cone_data.append((A_vals, A_rows, A_cols, b, K, []))
            self._variables.append(z)
            #
            # Linear inequalities
            #
            matrix = self.lifted_alpha[selector, :] - self.lifted_alpha[i, :]
            vecvar = self.lifted_mu_vars[i]
            A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar_minus_vecvar(-matrix, vecvar, z)
            num_rows = matrix.shape[0]
            b = np.zeros(num_rows)
            K = [Cone('+', num_rows)]
            A_rows = np.array(A_rows)
            cone_data.append((A_vals, A_rows, A_cols, b, K, []))
        #
        # the additional constraints, for the generalized AGE dual cone
        #
        mat = self.A
        vecvar = self.lifted_mu_vars[i]
        vec = self.b
        singlevar = self.v[i]
        A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar_plus_vec_times_singlevar(mat, vecvar, vec, singlevar)
        b = np.zeros(self.A.shape[0], )
        cur_K = [Cone(co.type, co.len) for co in self.K]
        cone_data.append((A_vals, A_rows, A_cols, b, cur_K, []))
        return cone_data

    def _dual_age_cone_violation(self, i, norm_ord, rough, v):
        from sageopt.coniclifts import clear_variable_indices, MIN as CL_MIN, SOLVED as CL_SOLVED
        selector = self.ech.expcovers[i]
        len_sel = np.count_nonzero(selector)
        if len_sel > 0:
            expr1 = np.tile(v[i], len_sel).ravel()
            expr2 = v[selector].ravel()
            lowerbounds = special_functions.rel_entr(expr1, expr2)
            mat = -(self.lifted_alpha[selector, :] - self.lifted_alpha[i, :])
            mu_i = self.lifted_mu_vars[i].value()
            # compute rough violation for this dual AGE cone
            residual = mat @ mu_i - lowerbounds
            residual[residual >= 0] = 0
            relent_viol = np.linalg.norm(residual, ord=norm_ord)
            AbK_val = self.A @ mu_i + v[i] * self.b
            AbK_viol = PrimalProductCone.project(AbK_val, self.K)
            curr_viol = relent_viol + AbK_viol
            # as applicable, solve an optimization problem to compute the violation.
            if curr_viol > 0 and not rough:
                temp_var = Variable(shape=(mat.shape[1],), name='temp_var')
                cons = [mat @ temp_var >= lowerbounds,
                        PrimalProductCone(self.A @ temp_var + v[i] * self.b, self.K)]
                prob = Problem(CL_MIN, Expression([0]), cons)
                status, value = prob.solve(verbose=False)
                clear_variable_indices()
                if status == CL_SOLVED and abs(value) < 1e-7:
                    curr_viol = 0
            return curr_viol
        else:
            return 0

    def variables(self):
        return self._variables

    def conic_form(self):
        if self.m > 1:
            nontrivial_I = list(set(self.ech.U_I + self.ech.P_I))
            con = self.v[nontrivial_I] >= 0
            con.epigraph_checked = True
            A_vals, A_rows, A_cols, b, K, _ = con.conic_form()
            cone_data = [(A_vals, A_rows, A_cols, b, K, [])]
            for i in self.ech.U_I:
                curr_age = self._dual_age_cone_data(i)
                cone_data += curr_age
            return cone_data
        else:
            con = self.v >= 0
            con.epigraph_checked = True
            A_vals, A_rows, A_cols, b, K, _ = con.conic_form()
            cone_data = [(A_vals, A_rows, A_cols, b, K, [])]
            return cone_data

    def violation(self, norm_ord=np.inf, rough=False):
        v = self.v.value()
        viols = []
        for i in self.ech.U_I:
            curr_viol = self._dual_age_cone_violation(i, norm_ord, rough, v)
            viols.append(curr_viol)
        viol = max(viols)
        return viol


class ExpCoverHelper(object):

    def __init__(self, alpha, c, A, b, K, expcovers=None):
        if c is not None and not isinstance(c, Expression):
            raise RuntimeError()
        self.alpha = alpha
        self.A = A
        self.b = b
        self.K = K
        self.m = alpha.shape[0]
        self.c = c
        if self.c is not None:
            self.U_I = [i for i, c_i in enumerate(self.c) if (not c_i.is_constant()) or (c_i.offset < 0)]
            # ^ indices of not-necessarily-positive sign; i \in U_I must get an AGE cone.
            # this AGE cone might not be used in the final solution to the associated
            # optimization problem.
            self.N_I = [i for i, c_i in enumerate(self.c) if (c_i.is_constant()) and (c_i.offset < 0)]
            # ^ indices of definitively-negative sign. if j \in N_I, then there is only one AGE
            # cone (indexed by i) with c^{(i)}_j != 0, and that's j == i. These AGE cones will
            # be used in any solution to the associated optimization problem, and we can be
            # certain that c^{(i)}_i == c_i.
            self.P_I = [i for i, c_i in enumerate(self.c) if (c_i.is_constant()) and (c_i.offset > 0)]
            # ^ indices of positive sign.
            # Together, union(self.P_I, self.U_I) comprise all indices "i" where c[i] is not identically zero.
        else:
            self.U_I = [i for i in range(self.m)]
            self.N_I = []
            self.P_I = []
        if isinstance(expcovers, dict):
            self._verify_exp_covers(expcovers)
        elif expcovers is None:
            expcovers = self._default_exp_covers()
        else:
            raise RuntimeError('Argument "expcovers" must be a dict.')
        self.expcovers = expcovers

    def _verify_exp_covers(self, expcovers):
        for i in self.U_I:
            if i not in expcovers:
                raise RuntimeError('Required key missing from "expcovers".')
            if (not isinstance(expcovers[i], np.ndarray)) or (expcovers[i].dtype != bool):
                raise RuntimeError('A value in "expcovers" was the wrong datatype.')
            if expcovers[i][i]:
                warnings.warn('Nonsensical value in "expcovers"; correcting.')
                expcovers[i][i] = False
        pass

    def _default_exp_covers(self):
        expcovers = dict()
        for i in self.U_I:
            cov = np.ones(shape=(self.m,), dtype=bool)
            cov[self.N_I] = False
            cov[i] = False
            expcovers[i] = cov
        if _AGGRESSIVE_REDUCTION_:
            row_sums = np.sum(self.alpha, 1)
            if np.all(self.alpha >= 0) and np.min(row_sums) == 0:
                # Then apply the reduction.
                zero_loc = np.nonzero(row_sums == 0)[0][0]
                for i in self.U_I:
                    if i == zero_loc:
                        continue
                    curr_cover = expcovers[i]
                    curr_row = self.alpha[i, :]
                    for j in range(self.m):
                        if curr_cover[j] and j != zero_loc and curr_row @ self.alpha[j, :] == 0:
                            curr_cover[j] = False
        return expcovers
