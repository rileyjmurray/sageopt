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
    Parameters
    ----------
    n : int
        The dimension of the space over which the constituent Signomials will be defined.

    Returns
    -------
    y : NumPy ndarray
        An array  of length ``n``, with ``y[i]`` as a Signomial with one term,
        corresponding to the (``i+1``)-th standard basis vector in ``n`` dimensional real space.

    Example
    -------
    This function is useful for constructing signomials in an algebraic form. ::

        y = standard_sig_monomials(3)
        f = (y[0] ** 1.5) * (y[2] ** -0.6) - y[1] * y[2]

    """
    y = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        ei = np.zeros(shape=(1, n))
        ei[0, i] = 1
        y[i] = Signomial(ei, np.array([1]))
    return y


class Signomial(object):
    """
    This class provides a symbolic representation for linear combinations of exponentials,
    composed with linear functions. These functions look like the following.

    .. math::

       x \mapsto \sum_{i=1}^m c_i \exp({\\alpha}_i \cdot x)

    The constructor for this class can be called in two different ways. The arguments
    to the constructor have names which reflect the different possibilities.

    Parameters
    ----------

    alpha_maybe_c : dict or NumPy ndarray

         If ``alpha_maybe_c`` is a dict, then it must be a dictionary from tuples-of-numbers to
         scalars. The keys will be converted to rows of a matrix which we call ``alpha``, and
         the values will be assembled into a vector which we call ``c``.

         If ``alpha_maybe_c`` is a NumPy ndarray, then the argument ``c`` must also be an ndarray,
         and ``c.size`` must equal ``alpha_maybe_c.shape[0]``.

    c : None or NumPy ndarray

        This value is only used when ``alpha_maybe_c`` is a NumPy ndarray. If that is the case, then
        this Signomial will represent the function ``lambda x: c @ np.exp(alpha_maybe_c @ x)``.

    Examples
    --------

    There are two ways to call the Signomial constructor.

    The first way is to specify a dictionary from tuples to scalars. The tuples are interpreted as linear
    functionals appearing in the exponential terms, and the scalars are the corresponding coefficients.::

        alpha_and_c = {(1,): 2}
        f = Signomial(alpha_and_c)
        print(f(1))  # equal to 2 * np.exp(1).
        print(f(0))  # equal to 2.

    The second way is to specify two arguments. In this case the first argument is a NumPy array
    where the rows represent linear functionals, and the second argument is a vector of corresponding
    coefficients.::

        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        c = np.array([1, 2, 3])
        f = Signomial(alpha, c)
        x = np.random.randn(2)
        print(f(x) - c @ np.exp(alpha @ x))  # zero, up to rounding errors.

    Attributes
    ----------

    n : int
        The dimension of the space over which this Signomial is defined. The number of columns in ``alpha``,
        and the length of tuples appearing in the dictionary ``alpha_c``.

    m : int
        The number of terms needed to express this Signomial in a basis of monomial functions
        ``lambda x: exp(a @ x)`` for row vectors ``a``. The signomial is presumed to include a constant term.

    alpha : NumPy ndarray
        Has shape ``(m, n)``. Each row specifies a vector appearing in an exponential function which
        defines this Signomial. The rows are ordered for consistency with the property ``c``.

    c : NumPy ndarray
        Has shape ``(m,)``. The scalar ``c[i]`` is this Signomial's coefficient on the basis function
        ``lambda x: exp(alpha[i, :] @ x)``. It is possible to have ``c.dtype == object``.

    alpha_c : defaultdict
        The keys of ``alpha_c`` are tuples of length ``n``, containing real numeric types (e.g int, float).
        These tuples define linear functions. This Signomial could be evaluated by the code snippet
        ``lambda x: np.sum([ alpha_c[a] * np.exp(a @ x) for a in alpha_c])``. The default value for this
        dictionary is zero.

    Notes
    -----

    Operator overloading.

        The Signomial class implements ``+``, ``-``, ``*``, and ``**`` between pairs of Signomials,
        and pairs involving one Signomial and one numeric type.

        The Signomial class also implements ``s1 / s2`` where ``s2`` is a numeric type or Signomial,
        but if ``s2`` is a Signomial, then its coefficient vector ``s2.c`` can only contain one nonzero entry.

        You can also use ``+``, ``-``, and ``*`` for pairs involving one Signomial and one non-Signomial.
        If ``s3`` is the result of such an operation, then ``s3.c`` will be a NumPy array with ``s3.dtype == object``.

    Function evaluations.

        Signomial objects are callable. If ``s`` is a Signomial object and ``x`` is a numpy array of length ``s.n``,
        then ``s(x)`` computes the Signomial object as though it were any other elementary Python function.

        Signomial objects provide functions to compute gradients (equivalently, Jacobians) and Hessians.
        These methods operate by caching and evaluating symbolic representations of the relevant partial derivatives.

    Internal representations.

        Both ``self.alpha_c`` and ``(self.alpha, self.c)`` completely specify a Signomial object.

        ``alpha_c`` is the dictionary which is taken to *define* this Signomial as a mathematical object.

        However, it is useful to have rapid access to the matrix of linear functionals ``alpha``, or the coefficient
        vector ``c`` as numpy arrays. So these fields are also maintained explicitly.

        If a user modifies the fields ``alpha`` or ``c`` directly, there may be an inconsistency between these
        fields, and the dictionary ``alpha_c``. Therefore the fields ``alpha`` and ``c`` should not be modified
        without taking great care to ensure consistency with the signomial's dictionary representation.
    """

    def __init__(self, alpha_maybe_c, c=None):
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
        self.alpha = None
        self.c = None
        self._grad = None
        self._hess = None
        self._update_alpha_c_arrays()
        pass

    def _cache_grad(self):
        if self._grad is None:
            g = np.empty(shape=(self.n,), dtype=object)
            for i in range(self.n):
                g[i] = self.partial(i)
            self._grad = g
        pass

    def _cache_hess(self):
        if self._hess is None:
            H = np.empty(shape=(self.n, self.n), dtype=object)
            for i in range(self.n):
                ith_partial = self.partial(i)
                for j in range(i+1):
                    curr_partial = ith_partial.partial(j)
                    H[i, j] = curr_partial
                    H[j, i] = curr_partial
            self._hess = H
        pass

    @property
    def grad(self):
        """
        A numpy ndarray of shape ``(n,)`` whose entries are Signomials. For a numpy ndarray ``x``, ``grad[i](x)``
        is the partial derivative of this Signomial with respect to coordinate ``i``, evaluated at ``x``.
        This array is constructed only when necessary, and is cached upon construction.
        """
        self._cache_grad()
        return self._grad


    @property
    def hess(self):
        """
        A numpy ndarray of shape ``(n, n)``, whose entries are Signomials. For a numpy ndarray ``x``,
        ``hess[i,j](x)`` is the (i,j)-th partial derivative of this Signomial, evaluated at ``x``.
        This array is constructed only when necessary, and is cached upon construction.
        """
        self._cache_hess()
        return self._hess

    def query_coeff(self, a):
        """
        Returns the coefficient of the basis function ``lambda x: np.exp(a @ x)`` for this Signomial.
        """
        tup = tuple(np.round(a, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__))
        if tup in self.alpha_c:
            return self.alpha_c[tup]
        else:
            return 0

    def constant_location(self):
        """
        Return the index ``i`` so that ``alpha[i, :]`` is the zero vector.
        """
        return np.where((self.alpha == 0).all(axis=1))[0][0]

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
        alpha1, c1 = self.alpha, self.c
        alpha2, c2 = other.alpha, other.c
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
        """
        Update ``alpha``, ``c``, and ``alpha_c`` to remove nonconstant terms where ``c[i] == 0``.
        """
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
        pass

    def partial(self, i):
        """
        Compute the symbolic partial derivative of this signomial, at coordinate ``i``.

        Parameters
        ----------
        i : int
            ``i`` must be an integer from 0 to ``self.n - 1``.

        Returns
        -------
        p : Signomial
            The function obtained by differentiating this signomial with respect to its i-th argument.
        """
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

    def grad_val(self, x):
        """
        Return the gradient of this Signomial (as a NumPy ndarray) at the point ``x``.
        """
        _ = self.grad  # construct the function handles.
        g = np.zeros(self.n)
        for i in range(self.n):
            g[i] = self._grad[i](x)
        return g

    def hess_val(self, x):
        """
        Return the Hessian of this Signomial (as a NumPy ndarray) at the point ``x``.
        """
        if self._hess is None:
            _ = self.hess  # ignore the return value
        H = np.zeros(shape=(self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                val = self._hess[i, j](x)
                H[i, j] = val
                H[j, i] = val
        return H

    def as_polynomial(self):
        """
        This function is only applicable if ``alpha`` is a matrix of nonnegative integers.

        Returns
        -------
        f : Polynomial
            For every elementwise vector ``x``, we have ``self(x) == f(np.exp(x))``.
        """
        from sageopt.symbolic.polynomials import Polynomial
        f = Polynomial(self.alpha, self.c)
        return f
