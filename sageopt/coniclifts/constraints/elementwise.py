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
from sageopt.coniclifts.constraints.constraint import Constraint
from sageopt.coniclifts.cones import Cone
import numpy as np


class ElementwiseConstraint(Constraint):

    _CURVATURE_CHECK_ = False

    _ELEMENTWISE_CONSTRAINT_ID_ = 0

    _ELEMENTWISE_OPS_ = ('==', '<=', '>=')

    def __init__(self, lhs, rhs, operator):
        from sageopt.coniclifts.base import Expression
        self.id = ElementwiseConstraint._ELEMENTWISE_CONSTRAINT_ID_
        ElementwiseConstraint._ELEMENTWISE_CONSTRAINT_ID_ += 1
        if not isinstance(lhs, Expression):
            lhs = Expression(lhs)
        if not isinstance(rhs, Expression):
            rhs = Expression(rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.initial_operator = operator
        name_str = 'Elementwise[' + str(self.id) + '] : '
        self.name = name_str
        if operator == '==':
            self.expr = (self.lhs - self.rhs).ravel()
            if ElementwiseConstraint._CURVATURE_CHECK_ and not self.expr.is_affine():
                raise RuntimeError('Equality constraints must be affine.')
            self.operator = '=='  # now we are a linear constraint "self.expr == 0"
            self.epigraph_checked = True
        else:  # elementwise inequality.
            if operator == '>=':
                self.expr = (self.rhs - self.lhs).ravel()
            else:
                self.expr = (self.lhs - self.rhs).ravel()
            if ElementwiseConstraint._CURVATURE_CHECK_ and not all(self.expr.is_convex()):
                raise RuntimeError('Cannot canonicalize.')
            self.operator = '<='  # now we are a convex constraint "self.expr <= 0"
            self.epigraph_checked = False

    def variables(self):
        """
        If this function is called before
        """
        all_vars = []
        all_vars_ids = set()
        # Find user-defined Variables
        lhs_vars = self.lhs.variables()
        for v in lhs_vars:
            if id(v) not in all_vars_ids:
                all_vars_ids.add(id(v))
                all_vars.append(v)
        rhs_vars = self.rhs.variables()
        for v in rhs_vars:
            if id(v) not in all_vars_ids:
                all_vars_ids.add(id(v))
                all_vars.append(v)
        # Look for epigraph variables
        expr_vars = self.expr.variables()
        for v in expr_vars:
            if id(v) not in all_vars_ids:
                all_vars_ids.add(id(v))
                all_vars.append(v)
        return all_vars

    def is_affine(self):
        if self.operator in ['==']:
            return True
        else:
            return self.expr.is_affine()

    def is_elementwise(self):
        return True

    def is_setmem(self):
        return False

    def conic_form(self):
        from sageopt.coniclifts.base import ScalarVariable
        # This function assumes that self.expr is affine (i.e. that any necessary epigraph
        # variables have been substituted into the nonlinear expression).
        #
        # The vector "K" returned by this function may only include entries for
        # the zero cone and R_+.
        #
        # Note: signs on coefficients are inverted in this function. This happens
        # because flipping signs on A and b won't affect the zero cone, and
        # it correctly converts affine constraints of the form "expression <= 0"
        # to the form "-expression >= 0". We want this latter form because our
        # primitive cones are the zero cone and R_+.
        if not self.epigraph_checked:
            raise RuntimeError('Cannot canonicalize without check for epigraph substitution.')
        m = self.expr.size
        b = np.empty(shape=(m,))
        if self.operator == '==':
            K = [Cone('0', m)]
        elif self.operator == '<=':
            K = [Cone('+', m)]
        else:
            raise RuntimeError('Unknown operator.')
        A_rows, A_cols, A_vals = [], [], []
        for i, se in enumerate(self.expr.flat):
            if len(se.atoms_to_coeffs) == 0:
                b[i] = -se.offset
                A_rows.append(i)
                A_cols.append(int(ScalarVariable.curr_variable_count())-1)
                A_vals.append(0)  # make sure scipy infers correct dimensions later on.
            else:
                b[i] = -se.offset
                A_rows += [i] * len(se.atoms_to_coeffs)
                col_idx_to_coeff = [(a.id, c) for a, c in se.atoms_to_coeffs.items()]
                A_cols += [atom_id for (atom_id, _) in col_idx_to_coeff]
                A_vals += [-c for (_, c) in col_idx_to_coeff]
        return A_vals, np.array(A_rows), A_cols, b, K

    def violation(self, norm_ord=None):
        expr = (self.lhs - self.rhs).as_expr()
        expr_val = expr.value
        if self.initial_operator == '<=':
            ignore = expr_val <= 0
            expr_val[ignore] = 0
        elif self.initial_operator == '>=':
            ignore = expr_val >= 0
            expr_val[ignore] = 0
        else:
            expr_val = np.abs(expr_val)
        residual = expr_val.ravel()
        viol = np.linalg.norm(residual, ord=norm_ord)
        return viol
