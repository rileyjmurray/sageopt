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
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.constraints.set_membership.conditional_sage_cone import PrimalCondSageCone
from sageopt.coniclifts.constraints.set_membership.ordinary_sage_cone import PrimalOrdinarySageCone
from sageopt.coniclifts.constraints.set_membership.conditional_sage_cone import DualCondSageCone
from sageopt.coniclifts.constraints.set_membership.ordinary_sage_cone import DualOrdinarySageCone


class PrimalSageCone(SetMembership):
    """


    Parameters
    ----------

    c : Expression

         The vector subject to this PrimalSageCone constraint.

    alpha : ndarray

        The rows of ``alpha`` are the exponents defining this primal SAGE cone.

    cond : tuple or None

        If None, then use an ordinary SAGE cone. If a tuple, then this constraint is a Conditional
        SAGE cone, and the tuple must be of the form ``cond = (A, b, K)``, specifying a feasible set
        in the coniclifts standard. The first ``alpha.shape[1]`` columns of ``A`` correspond (in order)
        to the variables over which the Signomial ``f = Signomial(alpha, c)`` would be defined.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts-standard.


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
        respect to ``alpha``, and ``c.value == sum([av.value for av in age_vectors.values()])``.

    cond : tuple or None

        If None, then use an ordinary SAGE cone. If a tuple, then this constraint is a Conditional
        SAGE cone, and the tuple must be of the form ``cond = (A, b, K)``, specifying a feasible set
        in the coniclifts standard. The first ``alpha.shape[1]`` columns of ``A`` correspond (in order)
        to the variables over which the Signomial ``f = Signomial(alpha, c)`` would be defined.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts standard.

    """

    def __init__(self, c, alpha, cond, name, **kwargs):
        covers = kwargs['covers'] if 'covers' in kwargs else None
        if cond is not None:
            raw_con = PrimalCondSageCone(c, alpha, cond, name, covers)
            self.cond = raw_con.AbK
        else:
            raw_con = PrimalOrdinarySageCone(c, alpha, name, covers)
            self.cond = None
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

    cond : tuple or None

        If None, then use an ordinary SAGE cone. If a tuple, then this constraint is a Conditional
        SAGE cone, and the tuple must be of the form ``cond = (A, b, K)``, specifying a feasible set
        in the coniclifts standard. The first ``alpha.shape[1]`` columns of ``A`` correspond (in order)
        to the variables over which the Signomial ``f = Signomial(alpha, c)`` would be defined.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts-standard.


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

    cond : tuple or None

        If None, then use an ordinary SAGE cone. If a tuple, then this constraint is a Conditional
        SAGE cone, and the tuple must be of the form ``cond = (A, b, K)``, specifying a feasible set
        in the coniclifts standard. The first ``alpha.shape[1]`` columns of ``A`` correspond (in order)
        to the variables over which the Signomial ``f = Signomial(alpha, c)`` would be defined.

    mu_vars : Dict[int, Variable]

        ``mu_vars[i]`` is the auxiliary variable associated with the i-th dual AGE cone.
        These variables are of shape ``mu_vars[i].size == self.n``. The most basic solution
        recovery algorithm takes these variables, and considers points ``x`` of the form
        ``x = mu_vars[i].value / self.v[i].value``.

    ech : ExpCoverHelper

        A simple wrapper around the constructor argument ``covers``. Manages validation of ``covers``
        when provided, and manages construction of ``covers`` when a user does not provide it.

    name : str

        Uniquely identifies this Constraint in the model where it appears. Serves as a suffix
        for the name of any auxiliary Variable created when compiling to the coniclifts-standard.
    """

    def __init__(self, v, alpha, cond, name, **kwargs):
        covers = kwargs['covers'] if 'covers' in kwargs else None
        c = kwargs['c'] if 'c' in kwargs else None
        if cond is not None:
            raw_con = DualCondSageCone(v, alpha, cond, name, c, covers)
            self.cond = raw_con.AbK
        else:
            raw_con = DualOrdinarySageCone(v, alpha, name, c, covers)
            self.cond = None
        self._raw_con = raw_con
        self.v = v
        self.alpha = alpha
        self.ech = raw_con.ech
        self.name = name
        self.mu_vars = raw_con.mu_vars
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








