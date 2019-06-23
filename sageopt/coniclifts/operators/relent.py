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
from sageopt.coniclifts.base import NonlinearScalarAtom, Expression, ScalarExpression, Variable
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
        self._id = RelEnt._REL_ENT_COUNTER_
        RelEnt._REL_ENT_COUNTER_ += 1
        self._args = (self.parse_arg(x), self.parse_arg(y))
        self.aux_var = None

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    def epigraph_conic_form(self):
        """
        Generate conic constraint for epigraph
            self.args[0] * ln( self.args[0] / self.args[1] ) <= self.aux_var.

        :return:
        """
        if self.aux_var is None:
            v = Variable(shape=(), name='_rel_ent_epi[' + str(self.id) + ']_')
            self.aux_var = v[()].scalar_variables()[0]
        b = np.zeros(3,)
        K = [Cone('e', 3)]
        # ^ initializations
        A_rows, A_cols, A_vals = [0], [self.aux_var.id], [-1]
        # ^ first row
        x = self.args[0]
        A_rows += (len(x)-1) * [2]
        A_cols += [var.id for var, co in x[:-1]]
        A_vals += [co for var, co in x[:-1]]
        b[2] = x[-1][1]
        # ^ third row
        y = self.args[1]
        A_rows += (len(y)-1) * [1]
        A_cols += [var.id for var, co in y[:-1]]
        A_vals += [co for var, co in y[:-1]]
        b[1] = y[-1][1]
        # ^ second row
        return A_vals, np.array(A_rows), A_cols, b, K, self.aux_var

    def value(self):
        vals = []
        for i in [0, 1]:
            arg_as_list = self.args[i]
            d = dict(arg_as_list[:-1])
            arg_se = ScalarExpression(d, arg_as_list[-1][1], verify=False)
            arg_val = arg_se.value()
            vals.append(arg_val)
        val = special_functions.rel_entr(vals[0], vals[1])
        return val
