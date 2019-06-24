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
import sageopt.coniclifts as cl
import numpy as np
from collections import defaultdict


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)

__EXPONENT_VECTOR_DECIMAL_POINTS__ = 7


def standard_sig_monomials(n):
    """
    Returns a numpy array "x" of length n, with "x[i]" as a signomial with
    one term corresponding to the (i+1)-th standard basis vector in R^n.

    This is useful for constructing signomials with syntax such as
        f = (x[0] ** 1.5) * (x[2] ** -0.6) - x[1] * x[2]
    """
    x = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        ei = np.zeros(shape=(1, n))
        ei[0, i] = 1
        x[i] = Signomial(ei, np.array([1]))
    return x


class Signomial(object):

    def __init__(self, alpha_maybe_c, c=None):
        """
        PURPOSE:

        Provide a symbolic representation of a function
            s(x) = \sum_{i=1}^m c[i] * exp(alpha[i] * x),
        where alpha[i] are row vectors of length n, and c[i] are scalars.

        CONSTRUCTION:

        There are two ways to call the Signomial constructor. The first way is to specify a dictionary from tuples to
        scalars. The tuples are interpreted as linear functionals alpha[i], and the scalars are the corresponding
        coefficients. The second way is to specify two arguments. In this case the first argument is a NumPy array
        where the rows represent linear functionals, and the second argument is a vector of corresponding coefficients.

        CAPABILITIES:

        Signomial objects are closed under addition, subtraction, and multiplication (but not division). These
        arithmetic operations are enabled through Python's default operator overloading conventions. That is,
            s1 + s2, s1 - s2, s1 * s2
        all do what you think they should. Arithmetic is defined between Signomial and non-Signomial objects by
        treating the non-Signomial object as a scalar; in such a setting the non-Signomial object is assumed
        to implement the operations "+", "-", and "*". Common use-cases include forming Signomials such as:
            s1 + v, s1 * v
        when "v" is a CVXPY Variable, CVXPY Expression, or numeric type.
        Signomial objects are callable. If x is a numpy array of length n, then s(x) computes the Signomial object
        as though it were any other elementary Python function.

        WORDS OF CAUTION:

        Signomials contain redundant information. In particular, s.alpha_c is the dictionary which is taken to *define*
        the signomial as a mathematical object. However, it is useful to have rapid access to the matrix of linear
        functionals "alpha", or the coefficient vector "c" as numpy arrays. The current implementation of this
        class is such that if a user modifies the variables s.c or s.alpha directly, there may be an inconsistency
        between these fields and the dictionary s.alpha_c. THEREFORE THOSE FIELDS SHOULD NOT BE MODIFIED WITHOUT TAKING
        GREAT CARE TO ENSURE CONSISTENCY WITH THE SIGNOMIAL'S DICTIONARY REPRESENTATION.

        PARAMETERS:

        :param alpha_maybe_c: either (1) a dictionary from tuples-of-numbers to scalars, or (2) a numpy array object
        with the same number of rows as c has entries (in the event that the second argument "c" is provided).
        :param c: optional. specified iff alpha_maybe_c is a numpy array.
        """
        # noinspection PyArgumentList
        if c is None:
            # noinspection PyArgumentList
            self.alpha_c = defaultdict(int, alpha_maybe_c)
        else:
            alpha = np.round(alpha_maybe_c, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
            alpha = alpha.tolist()
            if len(alpha) != c.size:
                raise RuntimeError('alpha and c specify different numbers of terms')
            self.alpha_c = defaultdict(int)
            for j in range(c.size):
                self.alpha_c[tuple(alpha[j])] += c[j]
        self.n = len(list(self.alpha_c.items())[0][0])
        self.alpha_c[(0,) * self.n] += 0  # ensures that there's a constant term.
        self.m = len(self.alpha_c)
        self.grad = None
        self.hess = None
        self._update_alpha_c_arrays()

    def constant_term(self):
        return self.alpha_c[(0,) * self.n]

    def query_coeff(self, a):
        """
        :param a: a numpy array of shape (self.n,).
        :return:
        """
        tup = tuple(a)  # convert from (likely) ndarray, to tuple.
        if tup in self.alpha_c:
            return self.alpha_c[tup]
        else:
            return 0

    def constant_location(self):
        return np.where((self.alpha == 0).all(axis=1))[0][0]

    def alpha_c_arrays(self):
        return self.alpha, self.c

    def _update_alpha_c_arrays(self):
        """
        Call this function whenever the dictionary representation of this Signomial object has been updated.
        """
        alpha = []
        c = []
        for k, v in self.alpha_c.items():
            alpha.append(k)
            c.append(v)
        self.alpha = np.array(alpha)
        self.c = np.array(c)
        if self.c.dtype is object:
            self.c = cl.Expression(self.c)

    def __add__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self.n
            d = {tup: other}
            other = Signomial(d)
        # noinspection PyArgumentList
        d = defaultdict(int, self.alpha_c)
        for k, v in other.alpha_c.items():
            d[k] += v
        res = Signomial(d)
        res.remove_terms_with_zero_as_coefficient()
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self.n
            d = {tup: other}
            other = Signomial(d)
        d = defaultdict(int)
        alpha1, c1 = self.alpha_c_arrays()
        alpha2, c2 = other.alpha_c_arrays()
        for i1, v1 in enumerate(alpha1):
            for i2, v2 in enumerate(alpha2):
                v3 = np.round(v1 + v2, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
                d[tuple(v3)] += c1[i1] * c2[i2]
        res = Signomial(d)
        res.remove_terms_with_zero_as_coefficient()
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_inv = other ** -1
        return self.__mul__(other_inv)

    def __rtruediv__(self, other):
        self_inv = self ** -1
        return self_inv * other

    def __sub__(self, other):
        # noinspection PyTypeChecker
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        # noinspection PyTypeChecker
        return other + (-1) * self

    def __pow__(self, power, modulo=None):
        if self.c.dtype not in __NUMERIC_TYPES__:
            raise RuntimeError('Cannot exponentiate signomials with symbolic coefficients.')
        if power % 1 == 0 and power >= 0:
            power = int(power)
            if power == 0:
                # noinspection PyTypeChecker
                return Signomial({(0,) * self.n: 1})
            else:
                s = Signomial(self.alpha_c)
                for t in range(power - 1):
                    s = s * self
                return s
        else:
            d = dict((k, v) for (k, v) in self.alpha_c.items() if v != 0)
            if len(d) != 1:
                raise ValueError('Only signomials with exactly one term can be raised to power %s.')
            v = list(d.values())[0]
            if v < 0 and not power % 1 == 0:
                raise ValueError('Cannot compute non-integer power %s of coefficient %s', power, v)
            alpha_tup = tuple(power * ai for ai in list(d.keys())[0])
            c = float(v) ** power
            s = Signomial(alpha_maybe_c={alpha_tup: c})
            return s

    def __neg__(self):
        # noinspection PyTypeChecker
        return self.__mul__(-1)

    def __call__(self, x):
        """
        Evaluates the mathematical function specified by the current Signomial object.

        :param x: either a scalar (if self.n == 1), or a numpy n-d array with len(x.shape) <= 2
        and x.shape[0] == self.n.
        :return:  If x is a scalar or an n-d array of shape (self.n,), then "val" is a numeric
        type equal to the signomial evaluated at x. If instead x is of shape (self.n, k) for
        some positive integer k, then "val" is a numpy n-d array of shape (k,), with val[i]
        equal to the current signomial evaluated on the i^th column of x.

        This function's behavior is undefined when x is not a scalar and has len(x.shape) > 2.
        """
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1 and x.ndim == 0):
            x = np.array([np.asscalar(np.array(x))])  # coerce into a 1d array of shape (1,).
        if not x.shape[0] == self.n:
            msg = 'Domain is R^' + str(self.n) + 'but x is in R^' + str(x.shape[0])
            raise ValueError(msg)
        if x.ndim > 2:
            msg = 'Signomials cannot be called on ndarrays with more than 2 dimensions.'
            raise ValueError(msg)
        x = x.astype(np.float128)
        exponents = np.dot(self.alpha.astype(np.float128), x)
        linear_vars = np.exp(exponents).astype(np.float128)
        val = np.dot(self.c, linear_vars)
        return val

    def __hash__(self):
        return hash(frozenset(self.alpha_c.items()))

    def __eq__(self, other):
        if not isinstance(other, Signomial):
            return False
        if self.m != other.m:
            return False
        for k in self.alpha_c:
            v = self.alpha_c[k]
            other_v = other.alpha_c[k]
            if not cl.Expression.are_equivalent(other_v, v, rtol=1e-8):
                return False
        return True

    def remove_terms_with_zero_as_coefficient(self):
        d = dict()
        for (k, v) in self.alpha_c.items():
            if (not isinstance(v, __NUMERIC_TYPES__)) or v != 0:
                d[k] = v
        # noinspection PyArgumentList
        self.alpha_c = defaultdict(int, d)
        tup = (0,) * self.n
        self.alpha_c[tup] += 0
        self.m = len(self.alpha_c)
        self._update_alpha_c_arrays()
        return

    def num_nontrivial_neg_terms(self):
        zero_location = self.constant_location()
        negs = np.where(self.c < 0)
        if len(negs[0]) > 0 and negs[0][0] == zero_location:
            return len(negs[0]) - 1
        else:
            return len(negs[0])

    def partial(self, i):
        if i < 0 or i >= self.n:
            raise RuntimeError('This Signomial does not have an input at index ' + str(i) + '.')
        d = defaultdict(int)
        for j in range(self.m):
            c = self.c[j] * self.alpha[j, i]
            if c != 0:
                vec = self.alpha[j, :].copy()
                vec[i] -= 1
                d[tuple(vec.tolist())] = c
        d[self.n * (0,)] += 0
        p = Signomial(d)
        return p

    def jac_val(self, x):
        if self.grad is None:
            g = np.empty(shape=(self.n,), dtype=object)
            for i in range(self.n):
                g[i] = self.partial(i)
            self.grad = g
        g = np.zeros(self.n)
        for i in range(self.n):
            g[i] = self.grad[i](x)
        return g

    def hess_val(self, x):
        if self.grad is None:
            self.jac_val(np.zeros(self.n))  # ignore return value
        if self.hess is None:
            H = np.empty(shape=(self.n, self.n), dtype=object)
            for i in range(self.n):
                ith_partial = self.partial(i)
                for j in range(i+1):
                    curr_partial = ith_partial.partial(j)
                    H[i, j] = curr_partial
                    H[j, i] = curr_partial
            self.hess = H
        H = np.zeros(shape=(self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                val = self.hess[i, j](x)
                H[i, j] = val
                H[j, i] = val
        return H

    def as_polynomial(self):
        from sageopt.symbolic.polynomials import Polynomial
        f = Polynomial(self.alpha, self.c)
        return f


