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
from sageopt.coniclifts.base import Variable, Expression
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.operators import affine as aff
from sageopt.coniclifts.operators.precompiled.relent import sum_relent, elementwise_relent
from sageopt.coniclifts.operators.precompiled import affine as compiled_aff
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.standards.constants import maximize as CL_MAX, solved as CL_SOLVED
import numpy as np
import scipy.special as special_functions
import warnings


class PrimalSageCone(SetMembership):

    def __init__(self, c, alpha, name, expcovers=None):
        self.name = name
        self.alpha = alpha
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.c = Expression(c)  # self.c is now definitely an ndarray of ScalarExpressions.
        self.ech = ExpCoverHelper(self.alpha, self.c, expcovers)
        self.nu_vars = dict()
        self.c_vars = dict()
        self.age_vectors = dict()
        self._variables = self.c.variables()
        self._initialize_primary_variables()
        pass

    def _initialize_primary_variables(self):
        if self.m > 2:
            for i in self.ech.U_I:
                nu_len = np.count_nonzero(self.ech.expcovers[i])
                if nu_len > 0:
                    self.nu_vars[i] = Variable(shape=(nu_len,), name='nu^{(' + str(i) + ')}_{' + self.name + '}')
                c_len = nu_len
                if i not in self.ech.N_I:
                    c_len += 1
                self.c_vars[i] = Variable(shape=(c_len,), name='c^{(' + str(i) + ')}_{' + self.name + '}')
            self._variables += list(self.nu_vars.values()) + list(self.c_vars.values())
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
            y = np.exp(1) * self.age_vectors[i][idx_set]  # This line consumes a large amount of runtime
            z = -self.age_vectors[i][i]
            name = self.name + '_' + str(i)
            A_vals, A_rows, A_cols, b, K, aux_vars = sum_relent(x, y, z, name)
        else:
            # just constrain -age_vectors[i][i] <= 0.
            A_vals, A_rows, A_cols = [1], np.array([0]), [self.age_vectors[i][i].scalar_variables()[0].id]
            b = np.array([0])
            K = [Cone('+', 1)]
            aux_vars = []
        if len(aux_vars) > 0:
            self._variables.append(aux_vars)
        return A_vals, A_rows, A_cols, b, K, []

    def _age_lin_eq_cone_data(self, i):
        # TODO: use the precompiled operator
        idx_set = self.ech.expcovers[i]
        matrix = (self.alpha[idx_set, :] - self.alpha[i, :]).T
        A_rows = np.tile(np.arange(matrix.shape[0]), reps=matrix.shape[1])
        ids = np.array(self.nu_vars[i].scalar_variable_ids)
        A_cols = np.repeat(ids, matrix.shape[0]).tolist()
        A_vals = matrix.ravel(order='F').tolist()  # stack columns, then tolist
        b = np.zeros(matrix.shape[0], )
        K = [Cone('0', matrix.shape[0])]
        return A_vals, A_rows, A_cols, b, K, []

    def _age_violation(self, i, norm_ord, c_i):
        if np.any(self.ech.expcovers[i]):
            idx_set = self.ech.expcovers[i]
            x_i = self.nu_vars[i].value
            x_i[x_i < 0] = 0
            y_i = np.exp(1) * c_i[idx_set]
            relent_res = np.sum(special_functions.rel_entr(x_i, y_i)) - c_i[i]  # <= 0
            relent_viol = abs(max(relent_res, 0))
            eq_res = (self.alpha[idx_set, :] - self.alpha[i, :]).T @ x_i  # == 0
            eq_res = eq_res.reshape((-1,))
            eq_viol = np.linalg.norm(eq_res, ord=norm_ord)
            total_viol = relent_viol + eq_viol
            return total_viol
        else:
            c_i = float(self.c_vars[i].value())  # >= 0
            return abs(min(0, c_i))

    def _age_vectors_sum_to_c(self):
        nonconst_locs = np.ones(self.m, dtype=bool)
        nonconst_locs[self.ech.N_I] = False
        aux_c_vars = list(self.age_vectors.values())
        aux_c_vars = aff.vstack(aux_c_vars).T
        aux_c_vars = aux_c_vars[nonconst_locs, :]
        main_c_var = self.c[nonconst_locs]
        A_vals, A_rows, A_cols, b = compiled_aff.columns_sum_to_vec(mat=aux_c_vars, vec=main_c_var)
        K = [Cone('0', b.size)]
        return A_vals, np.array(A_rows), A_cols, b, K, []

    def variables(self):
        return self._variables

    def conic_form(self):
        if self.m > 2:
            # Lift c_vars and nu_vars into Expressions of length self.m
            self._build_aligned_age_vectors()
            # Record all relative entropy constraints
            cone_data = []
            # age cones
            for i in self.ech.U_I:
                cone_data.append(self._age_rel_ent_cone_data(i))
                if np.any(self.ech.expcovers[i]):
                    cone_data.append(self._age_lin_eq_cone_data(i))
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
        c = self.c.value
        if self.m > 2:
            if not rough and c in self:
                return 0
            # compute violation for "AGE vectors sum to c"
            #   Although, we can use the fact that the SAGE cone contains R^m_++.
            #   and so only compute violation for "AGE vectors sum to <= c"
            age_vectors = {i: v.value for i, v in self.age_vectors.items()}
            sum_age_vectors = sum(age_vectors.values())
            residual = c - sum_age_vectors  # want >= 0
            residual[residual < 0] = 0
            sum_to_c_viol = np.linalg.norm(residual, ord=norm_ord)
            # compute violations for each AGE cone
            age_viols = np.zeros(shape=(len(self.ech.U_I,)))
            for idx, i in enumerate(self.ech.U_I):
                age_viols[idx] = self._age_violation(i, norm_ord, age_vectors[i])
            # add the max "AGE violation" to the violation for "AGE vectors sum to c".
            total_viol = sum_to_c_viol + np.max(age_viols)
            return total_viol
        else:
            residual = c.reshape((-1,))  # >= 0
            residual[residual >= 0] = 0
            return np.linalg.norm(c, ord=norm_ord)
        pass

    def __contains__(self, item):
        item = Expression(item)
        if item.is_constant() and np.all(item.value >= 0):
            return True
        con = PrimalSageCone(item, self.alpha, name='check_mem')
        prob = Problem(CL_MAX, Expression([0]), [con])
        status, value = prob.solve(verbose=False)
        from sageopt.coniclifts import clear_variable_indices
        clear_variable_indices()
        if status == CL_SOLVED:
            return abs(value) < 1e-7
        else:
            return False


class DualSageCone(SetMembership):

    def __init__(self, v, alpha, name, c=None, expcovers=None):
        """
        Aggregrates constraints on "v" so that "v" can be viewed as a dual variable
        to a constraint of the form "c \in C_{SAGE}(alpha)".

        :param v: a Coniclifts Expression of length s.m.
        """
        if c is None:
            self.c = None
        else:
            self.c = Expression(c)
        self.alpha = alpha
        self.ech = ExpCoverHelper(self.alpha, self.c, expcovers)
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.v = v
        self.name = name
        self.mu_vars = dict()
        self._variables = self.v.variables()
        self._initialize_primary_variables()
        pass

    def _initialize_primary_variables(self):
        if self.m > 2:
            for i in self.ech.U_I:
                self.mu_vars[i] = Variable(shape=(self.n,), name=('mu[' + str(i) + ']_{' + self.name + '}'))
                self._variables.append(self.mu_vars[i])
        pass

    def variables(self):
        return self._variables

    def conic_form(self):
        nontrivial_I = list(set(self.ech.U_I + self.ech.P_I))
        con = self.v[nontrivial_I] >= 0
        con.epigraph_checked = True
        A_vals, A_rows, A_cols, b, K, _ = con.conic_form()
        cone_data = [(A_vals, A_rows, A_cols, b, K, [])]
        if self.m > 2:
            for i in self.ech.U_I:
                curr_age = self._dual_age_cone_data(i)
                cone_data += curr_age
        return cone_data

    def _dual_age_cone_data(self, i):
        cone_data = []
        selector = self.ech.expcovers[i]
        len_sel = np.count_nonzero(selector)
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
        mat = self.alpha[selector, :] - self.alpha[i, :]
        A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar_minus_vecvar(-mat, self.mu_vars[i], z)
        num_rows = mat.shape[0]
        b = np.zeros(num_rows)
        K = [Cone('+', num_rows)]
        A_rows = np.array(A_rows)
        cone_data.append((A_vals, A_rows, A_cols, b, K, []))
        return cone_data

    def _dual_age_cone_violation(self, i, norm_ord, rough, v):
        from sageopt.coniclifts import clear_variable_indices
        selector = self.ech.expcovers[i]
        len_sel = np.count_nonzero(selector)
        expr1 = np.tile(v[i], len_sel).ravel()
        expr2 = v[selector].ravel()
        lowerbounds = special_functions.rel_entr(expr1, expr2)
        mat = -(self.alpha[selector, :] - self.alpha[i, :])
        vec = self.mu_vars[i].value()
        # compute rough violation for this dual AGE cone
        residual = mat @ vec - lowerbounds
        residual[residual >= 0] = 0
        curr_viol = np.linalg.norm(residual, ord=norm_ord)
        # as applicable, solve an optimization problem to compute the violation.
        if curr_viol > 0 and not rough:
            temp_var = Variable(shape=(mat.shape[1],), name='temp_var')
            prob = Problem(CL_MAX, Expression([0]), [mat @ temp_var >= lowerbounds])
            status, value = prob.solve(verbose=False)
            clear_variable_indices()
            if status == CL_SOLVED and abs(value) < 1e-7:
                curr_viol = 0
        return curr_viol

    def violation(self, norm_ord=np.inf, rough=False):
        v = self.v.value()
        viols = []
        for i in self.ech.U_I:
            curr_viol = self._dual_age_cone_violation(i, norm_ord, rough, v)
            viols.append(curr_viol)
        viol = max(viols)
        return viol

    def __contains__(self, item):
        from sageopt.coniclifts import clear_variable_indices
        if isinstance(item, Expression):
            item = item.value()
        for i in self.ech.U_I:
            selector = self.ech.expcovers[i]
            len_sel = np.count_nonzero(selector)
            expr1 = np.tile(item[i], len_sel)
            expr2 = item[selector]
            lowerbounds = special_functions.rel_entr(expr1, expr2).reshape((-1, 1))
            mat = -(self.alpha[selector, :] - self.alpha[i, :])
            temp_var = Variable(shape=(mat.shape[1], 1), name='temp_var')
            prob = Problem(CL_MAX, Expression([0]), [mat @ temp_var >= lowerbounds])
            status, value = prob.solve(verbose=False)
            clear_variable_indices()
            if status != CL_SOLVED or abs(value) > 1e-7:
                return False
        return True


class ExpCoverHelper(object):

    def __init__(self, alpha, c, expcovers=None):
        if c is not None and not isinstance(c, Expression):
            raise RuntimeError()
        self.alpha = alpha
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
        # Check if we are likely working with polynomials
        # (if so, go through the trouble of a reduction that
        #  applies to both polynomials and signomials, but
        #  is more likely to be useful in the former case.)
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
