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
import warnings
import scipy.special as special_functions


_ALLOWED_CONES_ = {'+', 'S', 'e', '0'}


SETTINGS = {
    'heuristic_reduction': True,
    'presolve_trivial_age_cones': False,
    'reduction_solver': 'ECOS',
    'sum_age_force_equality': False,
    'compact_dual': False
}


def check_cones(K):
    if any([co.type not in _ALLOWED_CONES_ for co in K]):
        raise NotImplementedError()
    pass


class PrimalSageCone(SetMembership):
    """
    Represent the constraint ":math:`c \\in C_{\\mathrm{SAGE}}(\\alpha, X)`".

    Parameters
    ----------

    c : Expression

         The vector subject to this PrimalSageCone constraint.

    alpha : ndarray

        The rows of ``alpha`` are the exponents defining this primal SAGE cone.

    X : SigDomain or None

        If None, then this constraint represents a primal ordinary-SAGE cone.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts standard.


    Other Parameters
    ----------------

    covers : Dict[int, ndarray]

        ``covers[i]`` indicates which indices ``j`` have ``alpha[j,:]`` participate in
        the i-th AGE cone. Automatically constructed in a presolve phase, if not provided.


    Attributes
    ----------

    alpha : ndarray

        The rows of ``alpha`` are the exponents defining this primal SAGE cone.

    c : Expression

        The vector subject to this PrimalSageCone constraint.

    age_vectors : Dict[int, Expression]

        If all Variable objects in the scope of this constraint are assigned feasible,
        values, then we should have ``age_vectors[i].value`` in the i-th AGE cone with
        respect to ``alpha``, and ``c.value == sum([av.value for av in age_vectors.values()], axis=0)``.

    X : SigDomain or None

        If None, then this constraint represents a primal ordinary-SAGE cone.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts standard.

    Notes
    -----

    The constructor can raise a RuntimeError if the constraint is deemed infeasible.

    """

    def __init__(self, c, alpha, X, name, **kwargs):
        self.settings = SETTINGS.copy()
        covers = kwargs['covers'] if 'covers' in kwargs else None
        if 'settings' in kwargs:
            self.settings.update(kwargs['settings'])
        self._n = alpha.shape[1]
        self._m = alpha.shape[0]
        self.name = name
        self.alpha = alpha
        self.X = X
        self.c = Expression(c)
        if X is not None:
            check_cones(X.K)
            self._lifted_n = X.A.shape[1]
            self.ech = ExpCoverHelper(self.alpha, self.c, (X.A, X.b, X.K), covers, self.settings)
        else:
            self._lifted_n = self._n
            self.ech = ExpCoverHelper(self.alpha, self.c, None, covers, self.settings)
        self.age_vectors = dict()
        self._nu_vars = dict()
        self._c_vars = dict()
        self._relent_epi_vars = dict()
        self._eta_vars = dict()
        self._initialize_variables()
        self._build_aligned_age_vectors()
        pass

    def _initialize_variables(self):
        self._variables = self.c.variables()
        if self._m > 1:
            for i in self.ech.U_I:
                num_cover = self.ech.expcover_counts[i]
                if num_cover > 0:
                    var_name = 'nu^{(' + str(i) + ')}_' + self.name
                    self._nu_vars[i] = Variable(shape=(num_cover,), name=var_name)
                    var_name = '_relent_epi_^{(' + str(i) + ')}_' + self.name
                    self._relent_epi_vars[i] = Variable(shape=(num_cover,), name=var_name)
                    if self.X is not None:
                        var_name = 'eta^{(' + str(i) + ')}_{' + self.name + '}'
                        self._eta_vars[i] = Variable(shape=(self.X.b.size,), name=var_name)
                c_len = num_cover
                if i not in self.ech.N_I:
                    c_len += 1
                if c_len == 0:  # pragma: no cover
                    msg = 'PrimalSageCone constraint "' + self.name + '" encountered an index '
                    msg += 'i=' + str(i) + '\n where the i-th AGE cone reduces to the nonnegative '
                    msg += 'orthant, but self.c[i]=' + str(self.c[i].value) + ' is negative. \n\n'
                    msg += 'This SAGE constraint is infeasible.'
                    raise RuntimeError(msg)
                var_name = 'c^{(' + str(i) + ')}_{' + self.name + '}'
                self._c_vars[i] = Variable(shape=(c_len,), name=var_name)
            self._variables += list(self._nu_vars.values())
            self._variables += list(self._c_vars.values())
            self._variables += list(self._relent_epi_vars.values())
            if self.X is not None:
                self._variables += list(self._eta_vars.values())
        pass

    def _build_aligned_age_vectors(self):
        if self._m > 1:
            for i in self.ech.U_I:
                ci_expr = Expression(np.zeros(self._m,))
                if i in self.ech.N_I:
                    ci_expr[self.ech.expcovers[i]] = self._c_vars[i]
                    ci_expr[i] = self.c[i]
                else:
                    ci_expr[self.ech.expcovers[i]] = self._c_vars[i][:-1]
                    ci_expr[i] = self._c_vars[i][-1]
                self.age_vectors[i] = ci_expr
        else:
            self.age_vectors[0] = self.c
        pass

    def _age_vectors_sum_to_c(self):
        nonconst_locs = np.ones(self._m, dtype=bool)
        nonconst_locs[self.ech.N_I] = False
        aux_c_vars = list(self.age_vectors.values())
        aux_c_vars = aff.vstack(aux_c_vars).T
        aux_c_vars = aux_c_vars[nonconst_locs, :]
        main_c_var = self.c[nonconst_locs]
        A_vals, A_rows, A_cols, b = comp_aff.columns_sum_leq_vec(aux_c_vars, main_c_var)
        conetype = '0' if self.settings['sum_age_force_equality'] else '+'
        K = [Cone(conetype, b.size)]
        return A_vals, A_rows, A_cols, b, K

    def variables(self):
        return self._variables

    def conic_form(self):
        if len(self._nu_vars) == 0:
            cd = self._trivial_conic_form()
        else:
            if self.X is None:
                cd = self._ordsage_conic_form()
            else:
                cd = self._condsage_conic_form()
        return cd

    def _trivial_conic_form(self):
        con = self.c >= 0
        con.epigraph_checked = True
        av, ar, ac, curr_b, curr_k = con.conic_form()
        cone_data = [(av, ar, ac, curr_b, curr_k)]
        return cone_data

    def _ordsage_conic_form(self):
        cone_data = []
        for i in self.ech.U_I:
            if i in self._nu_vars:
                idx_set = self.ech.expcovers[i]
                # relative entropy inequality constraint
                x = self._nu_vars[i]
                y = np.exp(1) * self.age_vectors[i][idx_set]  # This line consumes a large amount of runtime
                z = -self.age_vectors[i][i]
                epi = self._relent_epi_vars[i]
                cd = sum_relent(x, y, z, epi)
                cone_data.append(cd)
                # linear equality constraints
                mat = (self.alpha[idx_set, :] - self.alpha[i, :]).T
                av, ar, ac = comp_aff.mat_times_vecvar(mat, self._nu_vars[i])
                num_rows = mat.shape[0]
                curr_b = np.zeros(num_rows, )
                curr_k = [Cone('0', num_rows)]
                cone_data.append((av, ar, ac, curr_b, curr_k))
            else:
                con = 0 <= self.age_vectors[i][i]
                con.epigraph_checked = True
                cd = con.conic_form()
                cone_data.append(cd)
        cone_data.append(self._age_vectors_sum_to_c())
        return cone_data

    def _condsage_conic_form(self):
        cone_data = []
        lifted_alpha = self.alpha
        if self._lifted_n > self._n:
            zero_block = np.zeros(shape=(self._m, self._lifted_n - self._n))
            lifted_alpha = np.hstack((lifted_alpha, zero_block))
        for i in self.ech.U_I:
            if i in self._nu_vars:
                idx_set = self.ech.expcovers[i]
                # relative entropy inequality constraint
                x = self._nu_vars[i]
                y = np.exp(1) * self.age_vectors[i][idx_set]  # takes weirdly long amount of time.
                z = -self.age_vectors[i][i] + self._eta_vars[i] @ self.X.b
                epi = self._relent_epi_vars[i]
                cd = sum_relent(x, y, z, epi)
                cone_data.append(cd)
                # linear equality constraints
                mat1 = (lifted_alpha[idx_set, :] - lifted_alpha[i, :]).T
                mat2 = -self.X.A.T
                var1 = self._nu_vars[i]
                var2 = self._eta_vars[i]
                av, ar, ac = comp_aff.mat_times_vecvar_plus_mat_times_vecvar(mat1, var1, mat2, var2)
                num_rows = mat1.shape[0]
                curr_b = np.zeros(num_rows, )
                curr_k = [Cone('0', num_rows)]
                cone_data.append((av, ar, ac, curr_b, curr_k))
                # domain for "eta"
                con = DualProductCone(self._eta_vars[i], self.X.K)
                cone_data.extend(con.conic_form())
            else:
                con = 0 <= self.age_vectors[i][i]
                con.epigraph_checked = True
                cd = con.conic_form()
                cone_data.append(cd)
        cone_data.append(self._age_vectors_sum_to_c())
        return cone_data

    @staticmethod
    def project(item, alpha, X):
        if np.all(item >= 0):
            return 0
        c = Variable(shape=(item.size,))
        t = Variable(shape=(1,))
        cons = [
            vector2norm(item - c) <= t,
            PrimalSageCone(c, alpha, X, 'temp_con')
        ]
        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)
        return prob.value

    def violation(self, norm_ord=np.inf, rough=False):
        c = self.c.value
        if self._m > 1:
            if not rough:
                dist = PrimalSageCone.project(c, self.alpha, self.X)
                return dist
            # compute violation for "AGE vectors sum to c"
            #   Although, we can use the fact that the SAGE cone contains R^m_++.
            #   and so only compute violation for "AGE vectors sum to <= c"
            age_vectors = {i: v.value for i, v in self.age_vectors.items()}
            sum_age_vectors = sum(age_vectors.values())
            residual = c - sum_age_vectors  # want >= 0
            residual[residual > 0] = 0
            sum_to_c_viol = np.linalg.norm(residual, ord=norm_ord)
            # compute violations for each AGE cone
            alpha = self.alpha
            if self._lifted_n > self._n:
                # Then need to zero-pad alpha
                zero_block = np.zeros(shape=(self._m, self._lifted_n - self._n))
                alpha = np.hstack((alpha, zero_block))
            age_viols = []
            for i in self.ech.U_I:
                if i in self._nu_vars:
                    eta_vec = 0 if self.X is None else self._eta_vars[i].value
                    eta_viol = 0 if self.X is None else DualProductCone.project(eta_vec, self.X.K)
                    c_i = self.age_vectors[i].value
                    x_i = self._nu_vars[i].value
                    x_i[x_i < 0] = 0
                    idx_set = self.ech.expcovers[i]
                    y_i = np.exp(1) * c_i[idx_set]
                    condsage_term = 0 if self.X is None else self.X.b @ eta_vec
                    relent_res = np.sum(special_functions.rel_entr(x_i, y_i)) - c_i[i] + condsage_term  # <= 0
                    relent_viol = 0 if relent_res < 0 else relent_res
                    condsage_term = 0 if self.X is None else self.X.A.T @ eta_vec
                    eq_res = (alpha[idx_set, :] - alpha[i, :]).T @ x_i - condsage_term  # == 0
                    eq_res = eq_res.reshape((-1,))
                    eq_viol = np.linalg.norm(eq_res, ord=norm_ord)
                    total_viol = relent_viol + eq_viol + eta_viol
                    age_viols.append(total_viol)
                else:
                    c_i = float(self._c_vars[i].value)
                    relent_viol = 0 if c_i >= 0 else -c_i
                    age_viols.append(relent_viol)
            age_viols = np.array(age_viols)
            # add the max "AGE violation" to the violation for "AGE vectors sum to c".
            if np.any(age_viols == np.inf):
                total_viol = sum_to_c_viol + np.sum(age_viols[age_viols < np.inf])
                total_viol += PrimalSageCone.project(c, self.alpha, self.X)
            else:
                total_viol = sum_to_c_viol + np.max(age_viols)
            return total_viol
        else:
            residual = c.reshape((-1,))  # >= 0
            residual[residual >= 0] = 0
            return np.linalg.norm(c, ord=norm_ord)


class DualSageCone(SetMembership):
    """
    Represent the constraint ":math:`v \\in C_{\\mathrm{SAGE}}(\\alpha, X)^{\\dagger}`".

    Parameters
    ----------

    v : Expression

        The vector subject to the dual SAGE-cone constraint.

    alpha : ndarray

        The matrix of exponent vectors defining the SAGE cone; ``alpha.shape[0] == v.size``.

    X : SigDomain or None

        If None, then this constraint represents a dual ordinary-SAGE cone.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts standard.


    Other Parameters
    ----------------

    covers : Dict[int, ndarray]

        ``covers[i]`` indicates which indices ``j`` have ``alpha[j,:]`` participate in
        the i-th AGE cone. Automatically constructed in a presolve phase, if not provided.

    c : Expression or None

        When provided, this DualSageCone instance will compile to a constraint to ensure that ``v``
        is a valid dual variable to the constraint that :math:`c \\in C_{\\mathrm{SAGE}}(\\alpha, X)`,
        If we have have information about the sign of a component of  ``c``, then it is possible to
        reduce the number of coniclifts primitives needed to represent this constraint.


    Attributes
    ----------

    alpha : ndarray

        The rows of ``alpha`` are the exponents defining this primal SAGE cone.

    v : Expression

        The vector subject to this dual SAGE cone constraint.

    X : SigDomain

        If None, then this constraint represents a dual ordinary-SAGE cone.

    mu_vars : Dict[int, Variable]

        ``mu_vars[i]`` is the auxiliary variable associated with the i-th dual AGE cone.
        These variables are of shape ``mu_vars[i].size == alpha.shape[1]``. The most basic
        solution recovery algorithm takes these variables, and considers points ``x`` of
        the form ``x = mu_vars[i].value / self.v[i].value``.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts standard.
    """

    def __init__(self, v, alpha, X, name, **kwargs):
        self.settings = SETTINGS.copy()
        if 'settings' in kwargs:
            self.settings.update(kwargs['settings'])
        covers = kwargs['covers'] if 'covers' in kwargs else None
        c = kwargs['c'] if 'c' in kwargs else None
        self.alpha = alpha
        self.c = Expression(c) if c is not None else None
        self.v = v
        self._n = alpha.shape[1]
        self._m = alpha.shape[0]
        if X is not None:
            check_cones(X.K)
            self._lifted_n = X.A.shape[1]
            self.ech = ExpCoverHelper(self.alpha, self.c, (X.A, X.b, X.K), covers, self.settings)
        else:
            self._lifted_n = self._n
            self.ech = ExpCoverHelper(self.alpha, self.c, None, covers, self.settings)
        self.X = X
        self.mu_vars = dict()
        self.name = name
        self._lifted_mu_vars = dict()
        self._relent_epi_vars = dict()
        self._initialize_variables()
        pass

    def _initialize_variables(self):
        self._variables = self.v.variables()
        if self._m > 1:
            for i in self.ech.U_I:
                var_name = 'mu[' + str(i) + ']_{' + self.name + '}'
                self._lifted_mu_vars[i] = Variable(shape=(self._lifted_n,), name=var_name)
                self._variables.append(self._lifted_mu_vars[i])
                self.mu_vars[i] = self._lifted_mu_vars[i][:self._n]
                num_cover = self.ech.expcover_counts[i]
                if num_cover > 0 and not self.settings['compact_dual']:
                    var_name = '_relent_epi_[' + str(i) + ']_{' + self.name + '}'
                    epi = Variable(shape=(num_cover,), name=var_name)
                    self._relent_epi_vars[i] = epi
                    self._variables.append(epi)
        pass

    def variables(self):
        return self._variables

    def conic_form(self):
        if self._m > 1:
            nontrivial_I = list(set(self.ech.U_I + self.ech.P_I))
            con = self.v[nontrivial_I] >= 0
            # TODO: figure out when above constraint is implied by exponential cone constraints.
            con.epigraph_checked = True
            cd = con.conic_form()
            cone_data = [cd]
            for i in self.ech.U_I:
                idx_set = self.ech.expcovers[i]
                num_cover = self.ech.expcover_counts[i]
                if num_cover == 0:
                    continue
                expr = np.tile(self.v[i], num_cover).view(Expression)
                mat = self.alpha[i, :] - self.alpha[idx_set, :]
                vecvar = self._lifted_mu_vars[i][:self._n]
                if self.settings['compact_dual']:
                    epi = mat @ vecvar
                    cd = elementwise_relent(expr, self.v[idx_set], epi)
                    cone_data.append(cd)
                else:
                    # relative entropy constraints
                    epi = self._relent_epi_vars[i]
                    cd = elementwise_relent(expr, self.v[idx_set], epi)
                    cone_data.append(cd)
                    # Linear inequalities
                    av, ar, ac = comp_aff.mat_times_vecvar_minus_vecvar(mat, vecvar, epi)
                    num_rows = mat.shape[0]
                    curr_b = np.zeros(num_rows)
                    curr_k = [Cone('+', num_rows)]
                    cone_data.append((av, ar, ac, curr_b, curr_k))
                # membership in cone induced by self.AbK
                if self.X is not None:
                    A, b, K = self.X.A, self.X.b, self.X.K
                    vecvar = self._lifted_mu_vars[i]
                    singlevar = self.v[i]
                    av, ar, ac = comp_aff.mat_times_vecvar_plus_vec_times_singlevar(A, vecvar, b, singlevar)
                    curr_b = np.zeros(b.size, )
                    cone_data.append((av, ar, ac, curr_b, K))
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
            selector = self.ech.expcovers[i]
            num_cover = self.ech.expcover_counts[i]
            if num_cover > 0:
                expr1 = np.tile(v[i], num_cover).ravel()
                expr2 = v[selector].ravel()
                lowerbounds = special_functions.rel_entr(expr1, expr2)
                mat = -(self.alpha[selector, :] - self.alpha[i, :])
                mu_i = self._lifted_mu_vars[i].value
                # compute rough violation for this dual AGE cone
                residual = mat @ mu_i[:self._n] - lowerbounds
                residual[residual >= 0] = 0
                curr_viol = np.linalg.norm(residual, ord=norm_ord)
                if self.X is not None:
                    AbK_val = self.X.A @ mu_i + v[i] * self.X.b
                    AbK_viol = PrimalProductCone.project(AbK_val, self.X.K)
                    curr_viol += AbK_viol
                # as applicable, solve an optimization problem to compute the violation.
                if curr_viol > 0 and not rough:
                    temp_var = Variable(shape=(self._lifted_n,), name='temp_var')
                    cons = [mat @ temp_var[:self._n] >= lowerbounds]
                    if self.X is not None:
                        con = PrimalProductCone(self.X.A @ temp_var + v[i] * self.X.b, self.X.K)
                        cons.append(con)
                    prob = Problem(CL_MIN, Expression([0]), cons)
                    status, value = prob.solve(verbose=False)
                    if status == CL_SOLVED and abs(value) < 1e-7:
                        curr_viol = 0
                viols.append(curr_viol)
            else:
                viols.append(0)
        viol = max(viols)
        return viol


class ExpCoverHelper(object):

    def __init__(self, alpha, c, AbK, expcovers=None, settings=None):
        self.settings = SETTINGS.copy()
        if settings is not None:
            self.settings.update(settings)
        if c is not None and not isinstance(c, Expression):
            raise RuntimeError()
        self.m = alpha.shape[0]
        if AbK is not None:
            lifted_n = AbK[0].shape[1]
            n = alpha.shape[1]
            if lifted_n > n:
                # Then need to zero-pad alpha
                zero_block = np.zeros(shape=(self.m, lifted_n - n))
                alpha = np.hstack((alpha, zero_block))
        self.alpha = alpha
        self.AbK = AbK
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
        if self.AbK is None or self.settings['heuristic_reduction']:
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
                            """
                            The above operation is without loss of generality for ordinary SAGE
                            constraints. For conditional SAGE constraints, the operation may or
                            may-not be without loss of generality. As a basic check, the above
                            operation is w.l.o.g. even for conditional SAGE constraints, as long
                            as the "conditioning" satisfies the following property:

                                Suppose "y" a geometric-form solution which is feasible w.r.t.
                                conditioning. Then "y" remains feasible (w.r.t. conditioning)
                                when we assign "y[k] = 0".

                            The comments below explain in greater detail.

                            Observation
                            -----------
                            By being in this part of the code, there must exist a "k" where

                                 alpha[i,k] == 0 and alpha[j,k] > 0.

                            Also, we have alpha >= 0. These facts tell us that the expression

                                (alpha[j2,:] - alpha[i,:]) @ mu[:, i] (*)

                            is (1) non-decreasing in mu[k,i] for all 0 <= j2 < m, and (2)
                            strictly increasing in mu[k,i] when j2 == j. Therefore by
                            sending mu[i,k] to -\infty, we do not increase (*) for any
                            0 <= j2 < m, and in fact (*) goes to -\infty for j2 == j.

                            Consequence 1
                            -------------
                            If mu[:,i] is only subject to constraints of the form

                                v[i]*log(v[j2]/v[i]) >= (alpha[j2,:] - alpha[i,:]) @ mu[:, i]

                            with 0 <= j2 < m, then the particular constraint with j2 == j
                            is never active at any optimal solution. For ordinary SAGE cones,
                            this means the j-th term of alpha isn't used in the i-th AGE cone.

                            Consequence 2
                            -------------
                            For conditional SAGE cones, there is another constraint:

                                 A @ mu[:, i] + v[i] * b \in K.      (**)

                            However, as long as (**) allows us to send mu[k,i] to -\infty
                            without affecting feasibility, then the we arrive at the same
                            conclusion: the j-th term of alpha isn't used in the i-th AGE cone.
                            """
        if self.AbK is None:
            for i in self.U_I:
                if np.count_nonzero(expcovers[i]) == 1:
                    expcovers[i][:] = False
        if self.settings['presolve_trivial_age_cones']:
            if self.AbK is None:
                for i in self.U_I:
                    if np.any(expcovers[i]):
                        mat = self.alpha[expcovers[i], :] - self.alpha[i, :]
                        x = Variable(shape=(mat.shape[1],), name='temp_x')
                        objective = Expression([0])
                        cons = [mat @ x <= -1]
                        prob = Problem(CL_MIN, objective, cons)
                        prob.solve(verbose=False,
                                   solver=self.settings['reduction_solver'])
                        if prob.status == CL_SOLVED and abs(prob.value) < 1e-7:
                            expcovers[i][:] = False
            else:
                for i in self.U_I:
                    if np.any(expcovers[i]):
                        mat = self.alpha[expcovers[i], :] - self.alpha[i, :]
                        x = Variable(shape=(mat.shape[1],), name='temp_x')
                        t = Variable(shape=(1,), name='temp_t')
                        objective = t
                        A, b, K = self.AbK
                        cons = [mat @ x <= t, PrimalProductCone(A @ x + b, K)]
                        prob = Problem(CL_MIN, objective, cons)
                        prob.solve(verbose=False,
                                   solver=self.settings['reduction_solver'])
                        if prob.status == CL_SOLVED and prob.value < -100:
                            expcovers[i][:] = False
        return expcovers

