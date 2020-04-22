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


def weighted_sum_exp(c, x):
    """
    Return a coniclifts Expression of size 1, representing the signomial

        sum([ci * e^xi for (ci, xi) in (c, x)])

    :param c: a numpy ndarray of nonnegative numbers.
    :param x: a coniclifts Expression of the same size as c.
    """
    if not isinstance(x, Expression):
        x = Expression(x)
    if not isinstance(c, np.ndarray):
        c = np.array(c)
    if np.any(c < 0):
        raise RuntimeError('Epigraphs of non-constant signomials with negative terms are not supported.')
    if x.size != c.size:
        raise RuntimeError('Incompatible arguments.')
    x = x.ravel()
    c = c.ravel()
    kvs = []
    for i in range(x.size):
        if c[i] != 0:
            kvs.append((Exponential(x[i]), c[i]))
    d = dict(kvs)
    se = ScalarExpression(d, 0, verify=False)
    expr = se.as_expr()
    return expr


class Exponential(NonlinearScalarAtom):

    _EXPONENTIAL_COUNTER_ = 0

    @staticmethod
    def __atom_text__():
        return 'Exponential'

    def __init__(self, x):
        """
        Used to represent the epigraph of "e^x"
        :param x:
        """
        self._args = (self.parse_arg(x),)
        self._id = Exponential._EXPONENTIAL_COUNTER_
        Exponential._EXPONENTIAL_COUNTER_ += 1
        v = Variable(shape=(), name='_exp_epi_[' + str(self.id) + ']_')
        self._epigraph_variable = v[()].scalar_variables()[0]
        self._evaluator = Exponential._exp_evaluator
        pass

    @staticmethod
    def _exp_evaluator(vals):
        return np.exp(vals[0])

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    def epigraph_conic_form(self):
        """
        Refer to coniclifts/standards/cone_standards.txt to see that
        "(x, y) : e^x <= y" is represented as "(x, y, 1) \in K_{exp}".
        :return:
        """
        b = np.zeros(3,)
        K = [Cone('e', 3)]
        A_rows, A_cols, A_vals = [], [], []
        x = self.args[0]
        # first coordinate
        num_nonconst = len(x) - 1
        if num_nonconst > 0:
            A_rows += num_nonconst * [0]
            A_cols = [var.id for var, co in x[:-1]]
            A_vals = [co for var, co in x[:-1]]
        else:
            # infer correct dimensions later on
            A_rows.append(0)
            A_cols.append(ScalarVariable.curr_variable_count() - 1)
            A_vals.append(0)
        b[0] = x[-1][1]
        # second coordinate
        A_rows.append(1),
        A_cols.append(self._epigraph_variable.id)
        A_vals.append(1)
        # third coordinate (zeros for A, but included to infer correct dims later on)
        A_rows.append(2)
        A_cols.append(ScalarVariable.curr_variable_count() - 1)
        A_vals.append(0)
        b[2] = 1
        return A_vals, np.array(A_rows), A_cols, b, K


