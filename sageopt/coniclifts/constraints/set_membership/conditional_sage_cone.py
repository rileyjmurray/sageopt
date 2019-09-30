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
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.operators.precompiled.relent import sum_relent, elementwise_relent
from sageopt.coniclifts.operators.precompiled import affine as comp_aff
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.standards.constants import minimize as CL_MIN, solved as CL_SOLVED
import numpy as np
import warnings
from scipy.sparse import issparse
import scipy.special as special_functions

_ALLOWED_CONES_ = {'+', 'S', 'e', '0'}

_AGGRESSIVE_REDUCTION_ = True

_ELIMINATE_TRIVIAL_AGE_CONES_ = True

_REDUCTION_SOLVER_ = 'ECOS'


def check_cones(K):
    if any([co.type not in _ALLOWED_CONES_ for co in K]):
        raise NotImplementedError()
    else:
        newK = [Cone(co.type, co.len) for co in K]
        return newK


class PrimalCondSageCone(SetMembership):
    """
    This class assumes that the conic system {x : A @ x + b \in K } is feasible.
    """

    def __init__(self, c, alpha, X, name, covers=None):
        """

        Parameters
        ----------
        c
        alpha
        X : SigDomain
        name
        covers
        """
        self.name = name
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        K = check_cones(X.K)
        self.AbK = (X.A, X.b, K)
        self.X = X
        self.lifted_n = X.A.shape[1]
        if self.lifted_n > self.n:
            # Then need to zero-pad alpha
            zero_block = np.zeros(shape=(alpha.shape[0], self.lifted_n - self.n))
            alpha = np.hstack((alpha, zero_block))
        self.lifted_alpha = alpha
        self.c = Expression(c)  # self.c is now definitely an ndarray of ScalarExpressions.
        self.ech = ExpCoverHelper(self.lifted_alpha, self.c, self.AbK, covers)
        self.nu_vars = dict()
        self.c_vars = dict()
        self.relent_epi_vars = dict()
        self.age_vectors = dict()
        self.eta_vars = dict()
        self._variables = self.c.variables()
        self._initialize_variables()
        pass

    def _initialize_variables(self):
        if self.m > 1:
            for i in self.ech.U_I:
                num_cover = self.ech.expcover_counts[i]
                if num_cover > 0:
                    var_name = 'nu^{(' + str(i) + ')}_' + self.name
                    self.nu_vars[i] = Variable(shape=(num_cover,), name=var_name)
                    var_name = '_relent_epi_^{(' + str(i) + ')}_' + self.name
                    self.relent_epi_vars[i] = Variable(shape=(num_cover,), name=var_name)
                    # We assume the conic system is feasible. If num_cover == 0 and this is
                    # really w.l.o.g. (not some aggressive presolve), then it must be that
                    # the i-th AGE cone is the nonnegative orthant. Therefore only define
                    # eta_var[i] when num_cover > 0.
                    var_name = 'eta^{(' + str(i) + ')}_{' + self.name + '}'
                    self.eta_vars[i] = Variable(shape=(self.AbK[1].size,), name=var_name)
                c_len = num_cover
                if i not in self.ech.N_I:
                    c_len += 1
                var_name = 'c^{(' + str(i) + ')}_{' + self.name + '}'
                self.c_vars[i] = Variable(shape=(c_len,), name=var_name)
            self._variables += list(self.nu_vars.values())
            self._variables += list(self.c_vars.values())
            self._variables += list(self.relent_epi_vars.values())
            self._variables += list(self.eta_vars.values())
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

    def _age_violation(self, i, norm_ord, c_i, eta_i):
        # This is "rough" only.
        if self.ech.expcover_counts[i] > 0:
            A, b, K = self.AbK
            eta_viol = DualProductCone.project(eta_i, K)
            x_i = self.nu_vars[i].value
            x_i[x_i < 0] = 0
            idx_set = self.ech.expcovers[i]
            y_i = np.exp(1) * c_i[idx_set]
            relent_res = np.sum(special_functions.rel_entr(x_i, y_i)) - c_i[i] + b @ eta_i  # <= 0
            relent_viol = abs(max(relent_res, 0))
            eq_res = (self.lifted_alpha[idx_set, :] - self.lifted_alpha[i, :]).T @ x_i - A.T @ eta_i  # == 0
            eq_res = eq_res.reshape((-1,))
            eq_viol = np.linalg.norm(eq_res, ord=norm_ord)
            total_viol = relent_viol + eq_viol + eta_viol
            return total_viol
        else:
            c_i = float(self.c_vars[i].value)
            relent_viol = abs(min(0, c_i))
            return relent_viol

    def _age_vectors_sum_to_c(self):
        nonconst_locs = np.ones(self.m, dtype=bool)
        nonconst_locs[self.ech.N_I] = False
        aux_c_vars = list(self.age_vectors.values())
        aux_c_vars = aff.vstack(aux_c_vars).T
        aux_c_vars = aux_c_vars[nonconst_locs, :]
        main_c_var = self.c[nonconst_locs]
        A_vals, A_rows, A_cols, b = comp_aff.columns_sum_leq_vec(aux_c_vars, main_c_var)
        K = [Cone('+', b.size)]
        return A_vals, A_rows, A_cols, b, K

    def variables(self):
        return self._variables

    def conic_form(self):
        if self.m > 1:
            # Lift c_vars and nu_vars into Expressions of length self.m
            self._build_aligned_age_vectors()
            cone_data = []
            for i in self.ech.U_I:
                idx_set = self.ech.expcovers[i]
                if np.any(idx_set):
                    # relative entropy inequality constraint
                    x = self.nu_vars[i]
                    y = np.exp(1) * self.age_vectors[i][idx_set]  # takes weirdly long amount of time.
                    z = -self.age_vectors[i][i] + self.eta_vars[i] @ self.AbK[1]
                    epi = self.relent_epi_vars[i]
                    cd = sum_relent(x, y, z, epi)
                    cone_data.append(cd)
                    # linear equality constraints
                    mat1 = (self.lifted_alpha[idx_set, :] - self.lifted_alpha[i, :]).T
                    mat2 = -self.AbK[0].T
                    var1 = self.nu_vars[i]
                    var2 = self.eta_vars[i]
                    av, ar, ac = comp_aff.mat_times_vecvar_plus_mat_times_vecvar(mat1, var1, mat2, var2)
                    num_rows = mat1.shape[0]
                    curr_b = np.zeros(num_rows, )
                    curr_k = [Cone('0', num_rows)]
                    cone_data.append((av, ar, ac, curr_b, curr_k))
                    # domain for "eta"
                    con = DualProductCone(self.eta_vars[i], self.AbK[2])
                    cone_data.extend(con.conic_form())
                else:
                    con = 0 <= self.age_vectors[i][i]
                    con.epigraph_checked = True
                    cd = con.conic_form()
                    cone_data.append(cd)
            # Vectors sum to s.c
            cone_data.append(self._age_vectors_sum_to_c())
            return cone_data
        else:
            con = self.c >= 0
            con.epigraph_checked = True
            cd = con.conic_form()
            cone_data = [cd]
            return cone_data

    @staticmethod
    def project(item, alpha, X):
        if np.all(item >= 0):
            return 0
        c = Variable(shape=(item.size,))
        t = Variable(shape=(1,))
        cons = [
            vector2norm(item - c) <= t,
            PrimalCondSageCone(c, alpha, X=X, name='temp_con')
        ]
        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)
        return prob.value

    def violation(self, norm_ord=np.inf, rough=False):
        c = self.c.value
        if self.m > 1:
            if not rough:
                dist = PrimalCondSageCone.project(c, self.lifted_alpha, self.X)
                return dist
            # compute violation for "AGE vectors sum to c"
            #   Although, we can use the fact that the SAGE cone contains R^m_++.
            #   and so only compute violation for "AGE vectors sum to <= c"
            age_vectors = {i: v.value for i, v in self.age_vectors.items()}
            eta_vectors = {i: v.value for i, v in self.eta_vars.items()}
            sum_age_vectors = sum(age_vectors.values())
            residual = c - sum_age_vectors  # want >= 0
            residual[residual > 0] = 0
            sum_to_c_viol = np.linalg.norm(residual, ord=norm_ord)
            # compute violations for each AGE cone
            age_viols = np.zeros(shape=(len(self.ech.U_I,)))
            for idx, i in enumerate(self.ech.U_I):
                if i in eta_vectors:
                    eta_vec = eta_vectors[i]
                else:
                    eta_vec = None
                age_viols[idx] = self._age_violation(i, norm_ord, age_vectors[i], eta_vec)
            # add the max "AGE violation" to the violation for "AGE vectors sum to c".
            if np.any(age_viols == np.inf):
                total_viol = sum_to_c_viol + np.sum(age_viols[age_viols < np.inf])
                total_viol += PrimalCondSageCone.project(c, self.lifted_alpha, self.X)
            else:
                total_viol = sum_to_c_viol + np.max(age_viols)
            return total_viol
        else:
            residual = c.reshape((-1,))  # >= 0
            residual[residual >= 0] = 0
            return np.linalg.norm(c, ord=norm_ord)
        pass

    @property
    def alpha(self):
        return self.lifted_alpha[:, self.n]


class DualCondSageCone(SetMembership):
    """
    This class assumes that the conic system {x : A @ x + b \in K } is feasible.
    """

    def __init__(self, v, alpha, X, name, c=None, covers=None):
        """
        Aggregates constraints on "v" so that "v" can be viewed as a dual variable
        to a constraint of the form "c \in C_{SAGE}(alpha, A, b, K)".
        """
        K = check_cones(X.K)
        self.AbK = (X.A, X.b, K)
        self.X = X
        self.n = alpha.shape[1]
        self.m = alpha.shape[0]
        self.lifted_n = X.A.shape[1]
        if self.lifted_n > self.n:
            zero_block = np.zeros(shape=(alpha.shape[0], self.lifted_n - self.n))
            alpha = np.hstack((alpha, zero_block))
        self.lifted_alpha = alpha
        self.v = v
        self.name = name
        self._variables = self.v.variables()
        if c is None:
            self.c = None
        else:
            self.c = Expression(c)
        self.ech = ExpCoverHelper(self.lifted_alpha, self.c, self.AbK, covers)
        self.lifted_mu_vars = dict()
        self.mu_vars = dict()
        self.relent_epi_vars = dict()
        self._initialize_variables()
        pass

    def _initialize_variables(self):
        if self.m > 1:
            for i in self.ech.U_I:
                var_name = 'mu[' + str(i) + ']_{' + self.name + '}'
                self.lifted_mu_vars[i] = Variable(shape=(self.lifted_n,), name=var_name)
                self._variables.append(self.lifted_mu_vars[i])
                self.mu_vars[i] = self.lifted_mu_vars[i][:self.n]
                num_cover = self.ech.expcover_counts[i]
                if num_cover > 0:
                    var_name = '_relent_epi_[' + str(i) + ']_{' + self.name + '}'
                    epi = Variable(shape=(num_cover,), name=var_name)
                    self.relent_epi_vars[i] = epi
                    self._variables.append(epi)
        pass

    def _dual_age_cone_violation(self, i, norm_ord, rough, v):
        selector = self.ech.expcovers[i]
        num_cover = self.ech.expcover_counts[i]
        if num_cover > 0:
            expr1 = np.tile(v[i], num_cover).ravel()
            expr2 = v[selector].ravel()
            lowerbounds = special_functions.rel_entr(expr1, expr2)
            mat = -(self.lifted_alpha[selector, :] - self.lifted_alpha[i, :])
            mu_i = self.lifted_mu_vars[i].value
            # compute rough violation for this dual AGE cone
            residual = mat @ mu_i - lowerbounds
            residual[residual >= 0] = 0
            relent_viol = np.linalg.norm(residual, ord=norm_ord)
            A, b, K = self.AbK
            AbK_val = A @ mu_i + v[i] * b
            AbK_viol = PrimalProductCone.project(AbK_val, K)
            curr_viol = relent_viol + AbK_viol
            # as applicable, solve an optimization problem to compute the violation.
            if curr_viol > 0 and not rough:
                temp_var = Variable(shape=(mat.shape[1],), name='temp_var')
                cons = [mat @ temp_var >= lowerbounds,
                        PrimalProductCone(A @ temp_var + v[i] * b, K)]
                prob = Problem(CL_MIN, Expression([0]), cons)
                status, value = prob.solve(verbose=False)
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
            # TODO: figure out when above constraint is implied by exponential cone constraints.
            con.epigraph_checked = True
            cd = con.conic_form()
            cone_data = [cd]
            for i in self.ech.U_I:
                idx_set = self.ech.expcovers[i]
                num_cover = self.ech.expcover_counts[i]
                if num_cover > 0:
                    # relative entropy constraints
                    expr = np.tile(self.v[i], num_cover).view(Expression)
                    epi = self.relent_epi_vars[i]
                    cd = elementwise_relent(expr, self.v[idx_set], epi)
                    cone_data.append(cd)
                    # Linear inequalities
                    mat = self.lifted_alpha[idx_set, :] - self.lifted_alpha[i, :]
                    vecvar = self.lifted_mu_vars[i]
                    av, ar, ac = comp_aff.mat_times_vecvar_minus_vecvar(-mat, vecvar, epi)
                    num_rows = mat.shape[0]
                    curr_b = np.zeros(num_rows)
                    curr_k = [Cone('+', num_rows)]
                    cone_data.append((av, ar, ac, curr_b, curr_k))
                    # membership in cone induced by self.AbK
                    A, b, K = self.AbK
                    vecvar = self.lifted_mu_vars[i]
                    singlevar = self.v[i]
                    av, ar, ac = comp_aff.mat_times_vecvar_plus_vec_times_singlevar(A, vecvar, b, singlevar)
                    curr_b = np.zeros(b.size, )
                    curr_k = [Cone(co.type, co.len) for co in K]
                    cone_data.append((av, ar, ac, curr_b, curr_k))
            return cone_data
        else:
            con = self.v >= 0
            con.epigraph_checked = True
            cd = con.conic_form()
            cone_data = [cd]
            return cone_data

    def violation(self, norm_ord=np.inf, rough=False):
        v = self.v.value
        viols = []
        for i in self.ech.U_I:
            curr_viol = self._dual_age_cone_violation(i, norm_ord, rough, v)
            viols.append(curr_viol)
        viol = max(viols)
        return viol

    @property
    def alpha(self):
        return self.lifted_alpha[:, :self.n]


class ExpCoverHelper(object):

    def __init__(self, alpha, c, AbK, expcovers=None):
        if c is not None and not isinstance(c, Expression):
            raise RuntimeError()
        self.alpha = alpha
        self.AbK = AbK
        self.m = alpha.shape[0]
        self.c = c
        if self.c is not None:
            self.U_I = [i for i, c_i in enumerate(self.c) if (not c_i.is_constant()) or (c_i.offset < 0)]
            # ^ indices of not-necessarily-positive sign; i \in U_I must get an AGE cone.
            # this AGE cone might not be used in the final solution to the associated
            # optimization problem. These AGE cones might also be trivial (i.e. reduce to the nonnegative
            # orthant) if during presolve we set expcovers[i][:] = False.
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
        expcover_counts = {i: np.count_nonzero(expcovers[i]) for i in self.U_I}
        self.expcover_counts = expcover_counts

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
        if _ELIMINATE_TRIVIAL_AGE_CONES_:
            for i in self.U_I:
                if np.any(expcovers[i]):
                    mat = self.alpha[expcovers[i], :] - self.alpha[i, :]
                    x = Variable(shape=(mat.shape[1],), name='temp_x')
                    t = Variable(shape=(1,), name='temp_t')
                    objective = t
                    A, b, K = self.AbK
                    cons = [mat @ x <= t, PrimalProductCone(A @ x + b, K)]
                    prob = Problem(CL_MIN, objective, cons)
                    prob.solve(verbose=False, solver=_REDUCTION_SOLVER_)
                    if prob.status == CL_SOLVED and prob.value < -100:
                        expcovers[i][:] = False
        return expcovers
