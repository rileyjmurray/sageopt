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
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.operators.precompiled.relent import sum_relent, elementwise_relent
from sageopt.coniclifts.operators.precompiled import affine as compiled_aff
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.standards.constants import maximize as CL_MAX, solved as CL_SOLVED, minimize as CL_MIN
import numpy as np
import scipy.special as special_functions
import warnings


_ELIMINATE_TRIVIAL_AGE_CONES_ = True

_REDUCTION_SOLVER_ = 'ECOS'


class PrimalOrdinarySageCone(SetMembership):
    """
    Represent the constraint that a certain vector ``c`` belongs to the primal ordinary SAGE cone
    induced by a given set of exponent vectors ``alpha``. Maintain metadata such as summand
    "AGE vectors", and auxiliary variables needed to represent the primal SAGE cone in terms of
    coniclifts primitives. Instances of this class automatically apply a presolve procedure based
    on any constant components in ``c``, and geometric properties of the rows of ``alpha``.

    Parameters
    ----------

    c : Expression

        The vector subject to the primal SAGE-cone constraint.

    alpha : ndarray

        The matrix of exponent vectors defining the primal SAGE cone. ``alpha.shape[0] == c.size.``

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts-standard.

    covers : Dict[int, ndarray]

        ``covers[i]`` is a boolean selector array, indicating which exponents have a nontrivial role
        in representing the i-th AGE cone. A standard value for this argument is automatically
        constructed when unspecified. Providing this value can reduce the overhead associated
        with presolving a SAGE constraint.

    Attributes
    ----------

    alpha : ndarray

         The matrix whose rows define the exponent vectors of this primal SAGE cone.

    c : Expression

        The vector subject to the primal SAGE-cone constraint.

    age_vectors : Dict[int, Expression]

        ``age_vectors[i]`` is a lifted representation of ``c_vars[i]``. If ``c_vars`` and
        ``nu_vars`` are assigned feasible values, then we should have that ``age_vectors[i]``
        belongs to the i-th AGE cone induced by ``alpha``, and that
        ``self.c.value == np.sum([ av.value for av in age_vectors.values() ])``.

    m : int

        The number of rows in ``alpha``; the number of entries in ``c``.

    n : int

        The number of columns in ``alpha``.

    nu_vars : Dict[int, Variable]

        ``nu_vars[i]`` is an auxiliary Variable needed to represent the i-th AGE cone.
        The size of this variable is related to presolve behavior of ``self.ech``.

    c_vars : Dict[int, Variable]

        ``c_vars[i]`` is a Variable which determines the i-th summand in a SAGE decomposition
        of ``self.c``. The size of this variable is related to presolve behavior of ``self.ech``,
        and this can be strictly smaller than ``self.m``.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.
        This is an essential component of the duality relationship between PrimalOrdinarySageCone
        and DualOrdinarySageCone objects.
    """

    def __init__(self, c, alpha, name, covers=None):
        self.name = name
        self.alpha = alpha
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.c = Expression(c)  # self.c is now definitely an ndarray of ScalarExpressions.
        self.ech = ExpCoverHelper(self.alpha, self.c, covers)
        self.nu_vars = dict()
        self.c_vars = dict()
        self.relent_epi_vars = dict()
        self.age_vectors = dict()
        self._variables = self.c.variables()
        self._initialize_variables()
        pass

    def _initialize_variables(self):
        if self.m > 2:
            for i in self.ech.U_I:
                nu_len = np.count_nonzero(self.ech.expcovers[i])
                if nu_len > 0:
                    nu_i = Variable(shape=(nu_len,), name='nu^{(' + str(i) + ')}_{' + self.name + '}')
                    self.nu_vars[i] = nu_i
                    epi_i = Variable(shape=(nu_len,), name='_relent_epi_^{(' + str(i) + ')}_{' + self.name + '}')
                    self.relent_epi_vars[i] = epi_i
                c_len = nu_len
                if i not in self.ech.N_I:
                    c_len += 1
                self.c_vars[i] = Variable(shape=(c_len,), name='c^{(' + str(i) + ')}_{' + self.name + '}')
            self._variables += list(self.nu_vars.values())
            self._variables += list(self.c_vars.values())
            self._variables += list(self.relent_epi_vars.values())
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
            c_i = float(self.c_vars[i].value)  # >= 0
            return abs(min(0, c_i))

    def _age_vectors_sum_to_c(self):
        nonconst_locs = np.ones(self.m, dtype=bool)
        nonconst_locs[self.ech.N_I] = False
        aux_c_vars = list(self.age_vectors.values())
        aux_c_vars = aff.vstack(aux_c_vars).T
        aux_c_vars = aux_c_vars[nonconst_locs, :]
        main_c_var = self.c[nonconst_locs]
        A_vals, A_rows, A_cols, b = compiled_aff.columns_sum_leq_vec(mat=aux_c_vars, vec=main_c_var)
        K = [Cone('+', b.size)]
        return A_vals, np.array(A_rows), A_cols, b, K

    def conic_form(self):
        if self.m > 2:
            # Lift c_vars and nu_vars into Expressions of length self.m
            self._build_aligned_age_vectors()
            cone_data = []
            # age cones
            for i in self.ech.U_I:
                idx_set = self.ech.expcovers[i]
                if np.any(idx_set):
                    # relative entropy inequality constraint
                    x = self.nu_vars[i]
                    y = np.exp(1) * self.age_vectors[i][idx_set]  # This line consumes a large amount of runtime
                    z = -self.age_vectors[i][i]
                    epi = self.relent_epi_vars[i]
                    cd = sum_relent(x, y, z, epi)
                    cone_data.append(cd)
                    # linear equality constraints
                    mat = (self.alpha[idx_set, :] - self.alpha[i, :]).T
                    av, ar, ac = compiled_aff.mat_times_vecvar(mat, self.nu_vars[i])
                    num_rows = mat.shape[0]
                    curr_b = np.zeros(num_rows, )
                    curr_k = [Cone('0', num_rows)]
                    cone_data.append((av, ar, ac, curr_b, curr_k))
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
            A_vals, A_rows, A_cols, b, K = con.conic_form()
            cone_data = [(A_vals, A_rows, A_cols, b, K)]
            return cone_data

    def variables(self):
        return self._variables

    @staticmethod
    def project(item, alpha):
        if np.all(item >= 0):
            return 0
        c = Variable(shape=(item.size,))
        t = Variable(shape=(1,))
        cons = [
            vector2norm(item - c) <= t,
            PrimalOrdinarySageCone(c, alpha, 'temp_con')
        ]
        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)
        return prob.value

    def violation(self, norm_ord=np.inf, rough=False):
        c = self.c.value
        if self.m > 2:
            if not rough:
                dist = PrimalOrdinarySageCone.project(c, self.alpha)
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
            age_viols = np.zeros(shape=(len(self.ech.U_I,)))
            for idx, i in enumerate(self.ech.U_I):
                age_viols[idx] = self._age_violation(i, norm_ord, age_vectors[i])
            # add the max "AGE violation" to the violation for "AGE vectors sum to c".
            if np.any(age_viols == np.inf):
                total_viol = sum_to_c_viol + np.sum(age_viols[age_viols < np.inf])
                total_viol += PrimalOrdinarySageCone.project(c, self.alpha)
            else:
                total_viol = sum_to_c_viol + np.max(age_viols)
            return total_viol
        else:
            residual = c.reshape((-1,))  # >= 0
            residual[residual >= 0] = 0
            return np.linalg.norm(c, ord=norm_ord)
        pass


class DualOrdinarySageCone(SetMembership):
    """
    Represent the constraint that a certain vector ``v`` belongs to the dual ordinary SAGE
    cone induced by exponent vectors ``alpha``. Maintain auxiliary variables for each dual
    AGE cone (these auxiliary variables play a crucial role in recovering solutions from
    SAGE relaxations of signomial programs). Instances of this class automatically apply
    a presolve behavior based on:

        #. The geometric properties of the rows of ``alpha``, and

        #. Any constant components of an optional argument "``c``".

    Parameters
    ----------

    v : Expression

        The vector subject to the dual SAGE-cone constraint.

    alpha : ndarray

        The matrix of exponent vectors defining the SAGE cone; ``alpha.shape[0] == v.size``.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts-standard.

    c : Expression or None

        When provided, this DualOrdinarySageCone instance will compile to a constraint to ensure that ``v``
        is a valid dual variable to the constraint that :math:`c \\in C_{\\mathrm{SAGE}}(\\alpha)`.
        If ``c`` has some constant components, or we otherhave have information about the sign
        of a component of ``c``, then it is possible to reduce the number of coniclifts primitives
        needed to represent this constraint.

    covers : Dict[int, ndarray]

        ``covers[i]`` is a boolean selector array, indicating which exponents have a nontrivial role
        in representing the i-th AGE cone. A standard value for this argument is automatically
        constructed when unspecified. Providing this value can reduce the overhead associated
        with presolving a SAGE constraint.

    Attributes
    ----------

    m : int

        The number of rows in ``alpha``; the number of entries in ``v``.

    n : int

        The number of columns in ``alpha``.

    v : Expression

        The vector subject to the dual SAGE-cone constraint.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.
        This is an essential component of the duality relationship between PrimalSagecCone
        and DualOrdinarySageCone objects.

    c : Expression or None

        If not-None, this constraint will compile into primitives which only ensure that
        ``v`` is a valid dual variable to :math:`c \\in C_{\\mathrm{SAGE}}(\\alpha)`.

    mu_vars : Dict[int, Variable]

        ``mu_vars[i]`` is the auxiliary variable associated with the i-th dual AGE cone.
        These variables are of shape ``mu_vars[i].size == self.n``. The most basic solution
        recovery algorithm takes these variables, and considers points ``x`` of the form
        ``x = mu_vars[i].value / self.v[i].value``.

    """

    def __init__(self, v, alpha, name, c=None, covers=None):
        if c is None:
            self.c = None
        else:
            self.c = Expression(c)
        self.alpha = alpha
        self.ech = ExpCoverHelper(self.alpha, self.c, covers)
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.v = v
        self.name = name
        self.mu_vars = dict()
        self.relent_epi_vars = dict()
        self._variables = self.v.variables()
        self._initialize_variables()
        pass

    def _initialize_variables(self):
        if self.m > 2:
            for i in self.ech.U_I:
                num_cover = np.count_nonzero(self.ech.expcovers[i])
                if num_cover == 0:
                    continue

                var_name = 'mu[' + str(i) + ']_{' + self.name + '}'
                self.mu_vars[i] = Variable(shape=(self.n,), name=var_name)
                self._variables.append(self.mu_vars[i])

                var_name = '_relent_epi_[' + str(i) + ']_{' + self.name + '}'
                epi = Variable(shape=(num_cover,), name=var_name)
                self.relent_epi_vars[i] = epi
                self._variables.append(epi)
        pass

    def variables(self):
        return self._variables

    def conic_form(self):
        nontrivial_I = list(set(self.ech.U_I + self.ech.P_I))
        con = self.v[nontrivial_I] >= 0
        con.epigraph_checked = True
        A_vals, A_rows, A_cols, b, K = con.conic_form()
        cone_data = [(A_vals, A_rows, A_cols, b, K)]
        if self.m > 2:
            for i in self.ech.U_I:
                num_cover = np.count_nonzero(self.ech.expcovers[i])
                if num_cover == 0:
                    continue
                selector = self.ech.expcovers[i]
                len_sel = np.count_nonzero(selector)
                #
                # relative entropy constraints
                #
                expr1 = np.tile(self.v[i], len_sel).view(Expression)
                epi = self.relent_epi_vars[i]
                A_vals, A_rows, A_cols, b, K = elementwise_relent(expr1, self.v[selector], epi)
                cone_data.append((A_vals, A_rows, A_cols, b, K))
                #
                # Linear inequalities
                #
                mat = self.alpha[selector, :] - self.alpha[i, :]
                A_vals, A_rows, A_cols = compiled_aff.mat_times_vecvar_minus_vecvar(-mat, self.mu_vars[i], epi)
                num_rows = mat.shape[0]
                b = np.zeros(num_rows)
                K = [Cone('+', num_rows)]
                A_rows = np.array(A_rows)
                cone_data.append((A_vals, A_rows, A_cols, b, K))
        return cone_data

    def _dual_age_cone_violation(self, i, norm_ord, rough, v):
        selector = self.ech.expcovers[i]
        len_sel = np.count_nonzero(selector)
        if len_sel > 0:
            expr1 = np.tile(v[i], len_sel).ravel()
            expr2 = v[selector].ravel()
            lowerbounds = special_functions.rel_entr(expr1, expr2)
            mat = -(self.alpha[selector, :] - self.alpha[i, :])
            vec = self.mu_vars[i].value
            # compute rough violation for this dual AGE cone
            residual = mat @ vec - lowerbounds
            residual[residual >= 0] = 0
            curr_viol = np.linalg.norm(residual, ord=norm_ord)
            # as applicable, solve an optimization problem to compute the violation.
            if curr_viol > 0 and not rough:
                temp_var = Variable(shape=(mat.shape[1],), name='temp_var')
                prob = Problem(CL_MAX, Expression([0]), [mat @ temp_var >= lowerbounds])
                status, value = prob.solve(verbose=False)
                if status == CL_SOLVED and abs(value) < 1e-7:
                    curr_viol = 0
            return curr_viol
        else:
            residual = -v[i] if v[i] < 0 else 0
            return residual

    def violation(self, norm_ord=np.inf, rough=False):
        v = self.v.value
        viols = [0]
        for i in self.ech.U_I:
            num_cover = np.count_nonzero(self.ech.expcovers[i])
            if num_cover == 0:
                continue
            curr_viol = self._dual_age_cone_violation(i, norm_ord, rough, v)
            viols.append(curr_viol)
        viol = max(viols)
        return viol


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
        for i in self.U_I:
            if np.count_nonzero(expcovers[i]) == 1:
                expcovers[i][:] = False
        if _ELIMINATE_TRIVIAL_AGE_CONES_:
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
        for i in self.N_I:
            if np.count_nonzero(expcovers[i]) == 0:
                raise RuntimeError('This SAGE cone constraint is infeasible.')
        return expcovers
