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
from sageopt.coniclifts.base import NonlinearScalarAtom, Expression, ScalarExpression, Variable, ScalarVariable
from sageopt.coniclifts.cones import Cone


def vector2norm(x):
    if not isinstance(x, Expression):
        x = Expression(x)
    x = x.ravel()
    d = {Vector2Norm(x): 1}
    return ScalarExpression(d, 0).as_expr()


class Vector2Norm(NonlinearScalarAtom):

    _VECTOR_2_NORM_COUNTER_ = 0

    @staticmethod
    def __atom_text__():
        return 'Vector2Norm'

    def __init__(self, args):
        args = args.as_expr().ravel()
        self._args = tuple(self.parse_arg(v) for v in args)
        self._id = Vector2Norm._VECTOR_2_NORM_COUNTER_
        Vector2Norm._VECTOR_2_NORM_COUNTER_ += 1
        v = Variable(shape=(), name='_vec2norm_epi[' + str(self.id) + ']_')
        self._epigraph_variable = v[()].scalar_variables()[0]
        self._evaluator = Vector2Norm._vector2norm_evaluator

    @staticmethod
    def _vector2norm_evaluator(vals):
        vals = np.array(vals)
        res = np.linalg.norm(vals, ord=2)
        return res

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    def epigraph_conic_form(self):
        """
        Generate conic constraint for epigraph
            np.linalg.norm( np.array(self.args), ord=2) <= self._epigraph_variable
        The coniclifts standard for the second order cone (of length n) is
            { (t,x) : x \in R^{n-1}, t \in R, || x ||_2 <= t }.
        """
        m = len(self.args) + 1
        b = np.zeros(m,)
        A_rows, A_cols, A_vals = [0], [self._epigraph_variable.id], [1]  # for first row
        for i, arg in enumerate(self.args):
            nonconst_terms = len(arg) - 1
            if nonconst_terms > 0:
                A_rows += nonconst_terms * [i + 1]
                for var, coeff in arg[:-1]:
                    A_cols.append(var.id)
                    A_vals.append(coeff)
            else:
                # make sure we infer correct dimensions later on
                A_rows.append(i+1)
                A_cols.append(ScalarVariable.curr_variable_count() - 1)
                A_vals.append(0)
            b[i+1] = arg[-1][1]
        K = [Cone('S', m)]
        A_rows = np.array(A_rows)
        return A_vals, A_rows, A_cols, b, K




