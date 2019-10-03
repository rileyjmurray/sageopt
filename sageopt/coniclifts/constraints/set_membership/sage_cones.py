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
from sageopt.coniclifts.base import Expression
from sageopt.coniclifts.constraints.set_membership.conditional_sage_cone import PrimalCondSageCone
from sageopt.coniclifts.constraints.set_membership.ordinary_sage_cone import PrimalOrdinarySageCone
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
from scipy.sparse import issparse
import scipy.special as special_functions


_ALLOWED_CONES_ = {'+', 'S', 'e', '0'}

_AGGRESSIVE_REDUCTION_ = True

_ELIMINATE_TRIVIAL_AGE_CONES_ = True

_REDUCTION_SOLVER_ = 'ECOS'


def check_cones(K):
    if any([co.type not in _ALLOWED_CONES_ for co in K]):
        raise NotImplementedError()
    pass


#TODO: remove all references to ``cond``, and update references to ``X``.

class PrimalSageCone(SetMembership):
    """


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

    """

    def __init__(self, c, alpha, X, name, **kwargs):
        covers = kwargs['covers'] if 'covers' in kwargs else None
        if X is not None:
            raw_con = PrimalCondSageCone(c, alpha, X, name, covers)
        else:
            raw_con = PrimalOrdinarySageCone(c, alpha, name, covers)
        self.X = X
        self._raw_con = raw_con
        self.alpha = alpha
        self.c = Expression(c)
        self.age_vectors = raw_con.age_vectors
        self.ech = raw_con.ech
        self.name = name
        pass

    def variables(self):
        vs = self._raw_con.variables()
        return vs

    def conic_form(self):
        cf = self._raw_con.conic_form()
        return cf

    def violation(self, norm_ord=np.inf, rough=False):
        viol = self._raw_con.violation(norm_ord, rough)
        return viol


class DualSageCone(SetMembership):
    """


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
        where :math:`X` is determined by ``cond``. If we have have information about the sign of a
        component of  ``c``, then it is possible to reduce the number of coniclifts primitives
        needed to represent this constraint.


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
            self.ech = ExpCoverHelper(self.alpha, self.c, (X.A, X.b, X.K), covers)
        else:
            self._lifted_n = self._n
            self.ech = ExpCoverHelper(self.alpha, self.c, None, covers)
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
                if num_cover > 0:
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
                # relative entropy constraints
                expr = np.tile(self.v[i], num_cover).view(Expression)
                epi = self._relent_epi_vars[i]
                cd = elementwise_relent(expr, self.v[idx_set], epi)
                cone_data.append(cd)
                # Linear inequalities
                mat = self.alpha[idx_set, :] - self.alpha[i, :]
                vecvar = self._lifted_mu_vars[i][:self._n]
                av, ar, ac = comp_aff.mat_times_vecvar_minus_vecvar(-mat, vecvar, epi)
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

    def __init__(self, alpha, c, AbK, expcovers=None):
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
        if self.AbK is None or _AGGRESSIVE_REDUCTION_:
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
        if self.AbK is None:
            for i in self.U_I:
                if np.count_nonzero(expcovers[i]) == 1:
                    expcovers[i][:] = False
        if _ELIMINATE_TRIVIAL_AGE_CONES_:
            if self.AbK is None:
                for i in self.U_I:
                    if np.any(expcovers[i]):
                        mat = self.alpha[expcovers[i], :] - self.alpha[i, :]
                        x = Variable(shape=(mat.shape[1],), name='temp_x')
                        objective = Expression([0])
                        cons = [mat @ x <= -1]
                        prob = Problem(CL_MIN, objective, cons)
                        prob.solve(verbose=False, solver=_REDUCTION_SOLVER_)
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
                        prob.solve(verbose=False, solver=_REDUCTION_SOLVER_)
                        if prob.status == CL_SOLVED and prob.value < -100:
                            expcovers[i][:] = False
        return expcovers





