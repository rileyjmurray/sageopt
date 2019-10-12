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
import scipy.special as special_functions
from sageopt.coniclifts.base import NonlinearScalarAtom, Expression, ScalarExpression, Variable, ScalarVariable
from sageopt.coniclifts.utilities import array_index_iterator
from sageopt.coniclifts.cones import Cone


def relent(x, y, elementwise=False):
    if not isinstance(x, Expression):
        x = Expression(x)
    if not isinstance(y, Expression):
        y = Expression(y)
    if x.size != y.size:
        raise RuntimeError('Incompatible arguments to relent.')
    if elementwise:
        expr = np.empty(shape=x.shape, dtype=object)
        for tup in array_index_iterator(expr.shape):
            expr[tup] = ScalarExpression({RelEnt(x[tup], y[tup]): 1}, 0)
        return expr.view(Expression)
    else:
        x = x.ravel()
        y = y.ravel()
        d = dict((RelEnt(x[i], y[i]), 1) for i in range(x.size))
        return ScalarExpression(d, 0).as_expr()


class RelEnt(NonlinearScalarAtom):

    _REL_ENT_COUNTER_ = 0

    @staticmethod
    def __atom_text__():
        return 'RelEnt'

    def __init__(self, x, y):
        """
        Used to represent the epigraph of  "x * ln(x / y)".

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        """
        self._args = (self.parse_arg(x), self.parse_arg(y))
        self._id = RelEnt._REL_ENT_COUNTER_
        RelEnt._REL_ENT_COUNTER_ += 1
        v = Variable(shape=(), name='_rel_ent_epi[' + str(self.id) + ']_')
        self._epigraph_variable = v[()].scalar_variables()[0]
        self._evaluator = RelEnt._rel_entr_evaluator

    @staticmethod
    def _rel_entr_evaluator(vals):
        val = special_functions.rel_entr(vals[0], vals[1])
        return val

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    def epigraph_conic_form(self):
        """
        Generate conic constraint for epigraph
            self.args[0] * ln( self.args[0] / self.args[1] ) <= self._epigraph_variable.

        :return:
        """
        b = np.zeros(3,)
        K = [Cone('e', 3)]
        # ^ initializations
        A_rows, A_cols, A_vals = [0], [self._epigraph_variable.id], [-1]
        # ^ first row
        x = self.args[0]
        num_nonconst = len(x) - 1
        if num_nonconst > 0:
            A_rows += num_nonconst * [2]
            A_cols += [var.id for var, co in x[:-1]]
            A_vals += [co for var, co in x[:-1]]
        else:
            A_rows.append(2)
            A_cols.append(ScalarVariable.curr_variable_count() - 1)
            A_vals.append(0)
        b[2] = x[-1][1]
        # ^ third row
        y = self.args[1]
        num_nonconst = len(y) - 1
        if num_nonconst > 0:
            A_rows += num_nonconst * [1]
            A_cols += [var.id for var, co in y[:-1]]
            A_vals += [co for var, co in y[:-1]]
        else:
            A_rows.append(1)
            A_cols.append(ScalarVariable.curr_variable_count() - 1)
            A_vals.append(0)
        b[1] = y[-1][1]
        # ^ second row
        return A_vals, np.array(A_rows), A_cols, b, K


