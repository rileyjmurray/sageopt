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
from sageopt.coniclifts.cones import Cone


class PSD(SetMembership):

    _CURVATURE_CHECK_ = False

    _SYMMETRY_CHECK_ = True

    _CONSTRAINT_ID_ = 0

    _LMI_OPS_ = ('<<', '>>')

    def __init__(self, arg):
        self.id = PSD._CONSTRAINT_ID_
        PSD._CONSTRAINT_ID_ += 1
        self.arg = arg
        self.expr = None
        if PSD._SYMMETRY_CHECK_:
            from sageopt.coniclifts.base import Expression
            expr_sym = (arg + arg.T) / 2
            if not Expression.are_equivalent(arg, expr_sym):
                raise RuntimeError('Argument to LMI was not symmetric.')
        pass

    def variables(self):
        return self.arg.variables()

    def conic_form(self):
        from sageopt.coniclifts.base import Expression, ScalarVariable
        expr = np.triu(self.arg).view(Expression)
        expr = expr[np.triu_indices(expr.shape[0])]
        K = [Cone('P', expr.size)]
        b = np.empty(shape=(expr.size,))
        A_rows, A_cols, A_vals = [], [], []
        for i, se in enumerate(expr):
            b[i] = se.offset
            if len(se.atoms_to_coeffs) == 0:
                A_rows.append(i)
                A_cols.append(ScalarVariable.curr_variable_count())
                A_vals.append(0)  # make sure scipy infers correct dimensions later on.
            else:
                A_rows += [i] * len(se.atoms_to_coeffs)
                cols_and_coeff = [(a.id, c) for a, c in se.atoms_to_coeffs.items()]
                A_cols += [atom_id for (atom_id, _) in cols_and_coeff]
                A_vals += [c for (_, c) in cols_and_coeff]
        return [(A_vals, np.array(A_rows), A_cols, b, K)]
