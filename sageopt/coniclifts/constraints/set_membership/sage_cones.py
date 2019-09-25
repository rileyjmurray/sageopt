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
        respect to ``alpha``, and ``c.value == sum([av.value for av in age_vectors.values()])``.

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
        if X is not None:
            raw_con = DualCondSageCone(v, alpha, X, name, c, covers)
        else:
            raw_con = DualOrdinarySageCone(v, alpha, name, c, covers)
        self.X = X
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








