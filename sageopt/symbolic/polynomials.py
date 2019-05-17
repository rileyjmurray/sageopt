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
import sageopt.coniclifts as cl
from collections import defaultdict
from sageopt.symbolic.signomials import Signomial

__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def standard_poly_monomials(n):
    """
    Returns a numpy array "x" of length n, with "x[i]" as a Polynomial with
    one term corresponding to the (i+1)-th standard basis vector in R^n.

    This is useful for constructing polynomials with syntax such as ...

        x = standard_poly_monomials(3)

        f = (x[0] ** 2) * x[2] - 4 * x[1] * x[2] * x[0]
    """
    x = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        ei = np.zeros(shape=(1, n))
        ei[0, i] = 1
        x[i] = Polynomial(ei, np.array([1]))
    return x


class Polynomial(Signomial):

    def __init__(self, alpha_maybe_c, c=None):
        Signomial.__init__(self, alpha_maybe_c, c)
        if not np.all(self.alpha % 1 == 0):
            raise RuntimeError('Exponents must belong the the integer lattice.')
        if not np.all(self.alpha >= 0):
            raise RuntimeError('Exponents must be nonnegative.')
        self._sig_rep = None
        self._sig_rep_constrs = []

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            if isinstance(other, Signomial):
                raise RuntimeError('Cannot multiply signomials and polynomials.')
            # else, we assume that "other" is a scalar type
            other = Polynomial.promote_scalar_to_polynomial(other, self.n)
        self_var_coeffs = (self.c.dtype not in __NUMERIC_TYPES__)
        other_var_coeffs = (other.c.dtype not in __NUMERIC_TYPES__)
        if self_var_coeffs and other_var_coeffs:
            raise RuntimeError('Cannot multiply two polynomials that contain non-numeric coefficients.')
        temp = Signomial.__mul__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __truediv__(self, other):
        if not isinstance(other, __NUMERIC_TYPES__):
            raise RuntimeError('Cannot divide a polynomial by the non-numeric type: ' + type(other) + '.')
        other_inv = 1 / other
        return self.__mul__(other_inv)

    def __add__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):
            raise RuntimeError('Cannot add signomials to polynomials.')
        temp = Signomial.__add__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __sub__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):
            raise RuntimeError('Cannot subtract a signomial from a polynomial (or vice versa).')
        temp = Signomial.__sub__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __rmul__(self, other):
        # multiplication is commutative
        return Polynomial.__mul__(self, other)

    def __radd__(self, other):
        # addition is commutative
        return Polynomial.__add__(self, other)

    def __rsub__(self, other):
        # subtract self, from other
        # rely on correctness of __add__ and __mul__
        # noinspection PyTypeChecker
        return other + (-1) * self

    def __neg__(self):
        # rely on correctness of __mul__
        # noinspection PyTypeChecker
        return (-1) * self

    def __pow__(self, power, modulo=None):
        if self.c.dtype not in __NUMERIC_TYPES__:
            raise RuntimeError('Cannot exponentiate polynomials with symbolic coefficients.')
        temp = Signomial(self.alpha, self.c)
        temp = temp ** power
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    # noinspection PyMethodOverriding
    def __call__(self, x):
        """
        Numeric example:
            p = Polynomial({(1,2,3): -1})
            x = np.array([3,2,1])
            p(x) == (3 ** 1) * (2 ** 2) * (1 ** 3) * (-1) == -12

        Symbolic example:
            p = Polynomial({(2,): 1})
            x = Polynomial({(1,): 2, (0,): -1})
            w = p(x)
            w == Polynomial({(2,): 4, (1,): -4, (0,): 1}

        :param x: a scalar or an ndarray (of numeric types and/or Polynomials).
        :return: The polynomial "self" evaluated at x. If x is purely numeric, then this is
        a number. If x contains Polynomial objects, then the result is a Polynomial.

        NOTE: if "x" contains Polynomial entries, then those Polynomials must all be over the same
        number of variables. However, those Polynomials need not have the same number of variables
        as the current polynomial ("self").
        """
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1) or isinstance(x, Polynomial):
            # noinspection PyTypeChecker
            x = np.array([np.asscalar(np.array(x))])  # coerce into a 1d array of shape (1,).
        if not x.shape[0] == self.n:
            raise ValueError('The point must be in R^' + str(self.n) +
                             ', but the provided point is in R^' + str(x.shape[0]))
        if x.dtype == 'object':
            ns = np.array([xi.n for xi in x.flat if isinstance(xi, Polynomial)])
            if ns.size == 0:
                x = x.astype(np.float)
            elif np.any(ns != ns[0]):
                raise RuntimeError('The input vector cannot contain Polynomials over different variables.')
        if x.ndim == 1:
            temp1 = np.power(x, self.alpha)
            temp2 = np.prod(temp1, axis=1)
            val = np.dot(self.c, temp2)
            return val
        elif x.ndim == 2:
            vals = [self.__call__(xi) for xi in x.T]
            val = np.array(vals)
            return val
        else:
            raise ValueError('Can only evaluate on x with dimension <= 2.')

    def __hash__(self):
        return hash(frozenset(self.alpha_c.items()))

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return False
        if self.m != other.m:
            return False
        for k in self.alpha_c:
            v = self.alpha_c[k]
            other_v = other.alpha_c[k]
            if not cl.Expression.are_equivalent(other_v, v, rtol=1e-8):
                return False
        return True

    def even_locations(self):
        evens = np.where(~(self.alpha % 2).any(axis=1))[0]
        return evens

    def partial(self, i):
        if i < 0 or i >= self.n:
            raise RuntimeError('This polynomial does not have an input at index ' + str(i) + '.')
        d = defaultdict(int)
        for j in self.m:
            if self.alpha[j, i] > 0:
                vec = self.alpha[j, :].copy()
                c = self.c[j] * vec[i]
                vec[i] -= 1
                d[tuple(vec.tolist())] = c
        d[self.n * (0,)] += 0
        p = Polynomial(d)
        return p

    def standard_multiplier(self):
        evens = self.even_locations()
        mult_alpha = self.alpha[evens, :].copy()
        mult_c = np.ones(len(evens))
        mult = Polynomial(mult_alpha, mult_c)
        if mult.alpha_c[self.n * (0,)] == 0:
            mult += 1
        return mult

    @property
    def sig_rep(self):
        # It is important that self._sig_rep.alpha == self.alpha.
        if self._sig_rep is None:
            self.compute_sig_rep()
        return self._sig_rep, self._sig_rep_constrs

    def compute_sig_rep(self):
        self._sig_rep = None
        self._sig_rep_constrs = []
        sigrep_c = np.zeros(shape=(self.m,), dtype=object)
        need_vars = []
        for i, row in enumerate(self.alpha):
            if np.any(row % 2 != 0):
                if isinstance(self.c[i], __NUMERIC_TYPES__):
                    sigrep_c[i] = -abs(self.c[i])
                else:
                    need_vars.append(i)
            else:
                if isinstance(self.c[i], np.ndarray):
                    sigrep_c[i] = self.c[i][()]
                else:
                    sigrep_c[i] = self.c[i]
        if len(need_vars) > 0:
            var_name = str(self) + ' variable sigrep coefficients'
            c_hat = cl.Variable(shape=(len(need_vars),), name=var_name)
            sigrep_c[need_vars] = c_hat
            self._sig_rep_constrs.append(c_hat <= self.c[need_vars])
            self._sig_rep_constrs.append(c_hat <= -self.c[need_vars])
        self._sig_rep = Signomial(self.alpha, sigrep_c)
        pass

    @staticmethod
    def promote_scalar_to_polynomial(scalar, n):
        alpha = np.array([[0] * n])
        c = np.array([scalar])
        return Polynomial(alpha, c)
