"""
   Copyright 2020 Riley John Murray

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
from sageopt.coniclifts.base import NonlinearScalarAtom, Expression, ScalarExpression, Variable
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.utilities import array_index_iterator


def pos(x, eval_only=False):
    """
    Return a coniclifts Expression representing max(x,0) componentwise.

    :param x: a coniclifts Expression.

    :param eval_only: bool. True if the returned Expression will not be used
    in an optimization problem.
    """
    if not isinstance(x, Expression):
        x = Expression(x)
    expr = np.empty(shape=x.shape, dtype=object)
    for tup in array_index_iterator(expr.shape):
        expr[tup] = ScalarExpression({Pos(x[tup], eval_only): 1}, 0, verify=False)
    return expr.view(Expression)


class Pos(NonlinearScalarAtom):

    _POS_COUNTER_ = 0

    @staticmethod
    def __atom_text__():
        return 'Pos'

    def __init__(self, x, eval_only=False):
        """
        Used to represent the epigraph of "max(0, x)"
        :param x: Expression-like of size 1.
        """
        self._args = (self.parse_arg(x),)
        self._id = Pos._POS_COUNTER_
        Pos._POS_COUNTER_ += 1
        self._epigraph_variable = None
        self._eval_only = eval_only
        if not eval_only:
            v = Variable(shape=(), name='_pos_epi_[' + str(self.id) + ']_')
            self._epigraph_variable = v[()].scalar_variables()[0]
        self._evaluator = Pos._pos_evaluator
        pass

    @staticmethod
    def _pos_evaluator(vals):
        return max(vals[0], 0)

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    def epigraph_conic_form(self):
        """
        max(0,x) <= epi is represented as 0 <= epi,  x <= epi.
        """
        b = np.zeros(2,)
        K = [Cone('+', 2)]
        A_rows = [0, 1]
        A_cols = [self._epigraph_variable.id, self._epigraph_variable.id]
        A_vals = [1, 1]
        x = self.args[0]
        num_nonconst = len(x) - 1
        if num_nonconst > 0:
            A_rows += num_nonconst * [1]
            A_cols += [var.id for var, co in x[:-1]]
            A_vals += [-co for var, co in x[:-1]]
        # b[0] = 0; already set.
        b[1] = -x[-1][1]
        return A_vals, np.array(A_rows), A_cols, b, K


