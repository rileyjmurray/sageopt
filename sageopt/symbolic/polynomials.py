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
    Parameters
    ----------
    n : int
        The dimension of the space over which the constituent Polynomials will be defined.

    Returns
    -------
    x : NumPy ndarray
        An array  of length ``n``. ``x[i]`` is the Polynomial that evaluates to ``x[i](z) = z[i]`` for all numpy
        ndarrays ``z`` of length ``n``.

    Example
    -------
    This function is useful for constructing Polynomials in an algebraic form. ::

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
    """
    This class provides a symbolic representation for polynomials which are sparse in the monomial basis.
    These functions look like the following.

    .. math::

       x \mapsto \sum_{i=1}^m c_i \prod_{j=1}^n {x_j}^{\\alpha_{ij}}

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
        this Polynomial will represent the function ``lambda x: c @ np.prod(np.power(alpha_maybe_c, x))``.

    Examples
    --------

    There are two ways to call the Polynomial constructor.

    The first way is to specify a dictionary from tuples to scalars. Each key-value pair of the dictionary
    represents a monomial appearing in this polynomial. The key tuples must all be of some common length ``n``.
    If ``(a, coeff)`` is a key-value pair in this dictionary, then the Polynomial includes an additive term
    ``lambda x: coeff * np.prod(np.power(a, x))``. ::

        alpha_and_c = {(1,): 2}
        f = Polynomial(alpha_and_c)
        print(f(1))  # equal to 2.
        print(f(-3))  # equal to -6.

    The second way is to specify two arguments. In this case the first argument ``alpha`` is a NumPy array of
    exponent vectors, where ``alpha[i, j]`` is the power of variable ``j`` in monomial ``i``. The second argument
    is a numpy array ``c``, where ``c[i]`` is the coefficient on the i-th monomial defined by ``alpha``. ::

        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        c = np.array([1, 2, 3])
        f = Polynomial(alpha, c)
        x = np.array([-4, 7])
        val = f(x)  # val = 1 * (-4) + 2 * (7) + 3 * (-4 * 7)
        print(val)  # -74

    Attributes
    ----------

    n : int
        The dimension of the space over which this Polynomial is defined. The number of columns in ``alpha``,
        and the length of tuples appearing in the dictionary ``alpha_c``.

    m : int
        The number of terms needed to express this Polynomial in the standard monomial basis.
        The number of rows in  ``alpha``.  The length of the dictionary ``alpha_c``.

    alpha : NumPy ndarray
        Has shape ``(m, n)``. Entry ``alpha[i,j]`` is the power of an implicit variable ``x[j]``
        appearing in the i-th monomial for this Polynomial. The i-th monomial, in turn, has coefficient ``c[i]``.

    c : NumPy ndarray
        Has shape ``(m,)``. The scalar ``c[i]`` is this Polynomial's coefficient on the basis function
        ``lambda x: np.prod(np.power(alpha[i, :], x))``. It is possible to have ``c.dtype == object``, to allow for
        coniclifts Variables.

    alpha_c : defaultdict
        The keys of ``alpha_c`` are tuples of length ``n``, containing real numeric types (e.g int, float).
        These tuples correspond to rows in ``alpha``.

    Notes
    -----
    The Polynomial class subclasses the Signomial class. This is done because most algebraic operations between
    polynomials and signomials are identical. However it is important to remember that polynomials and signomials
    evaluate in very different ways.

    """

    def __init__(self, alpha_maybe_c, c=None):
        Signomial.__init__(self, alpha_maybe_c, c)
        if not np.all(self.alpha % 1 == 0):
            raise RuntimeError('Exponents must belong the the integer lattice.')
        if not np.all(self.alpha >= 0):
            raise RuntimeError('Exponents must be nonnegative.')
        self._sig_rep = None
        self._sig_rep_constrs = []
        pass

    @property
    def grad(self):
        """
        A numpy ndarray of shape ``(n,)`` whose entries are Polynomials. For a numpy ndarray ``x``,
        ``grad[i](x)`` is the partial derivative of this Polynomial with respect to coordinate ``i``,
        evaluated at ``x``. This array is constructed only when necessary, and is cached upon construction.
        """
        Signomial._cache_grad(self)
        return self._grad

    @property
    def hess(self):
        """
        A numpy ndarray of shape ``(n, n)``, whose entries are Polynomials. For a numpy ndarray ``x``,
        ``hess[i,j](x)`` is the (i,j)-th partial derivative of this Polynomial, evaluated at ``x``.
        This array is constructed only when necessary, and is cached upon construction.
        """
        Signomial._cache_hess(self)
        return self._hess

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            if isinstance(other, Signomial):
                raise RuntimeError('Cannot multiply signomials and polynomials.')
            # else, we assume that "other" is a scalar type
            tup = (0,) * self.n
            d = {tup: other}
            other = Polynomial(d)
        self_var_coeffs = (self.c.dtype not in __NUMERIC_TYPES__)
        other_var_coeffs = (other.c.dtype not in __NUMERIC_TYPES__)
        if self_var_coeffs and other_var_coeffs:
            raise RuntimeError('Cannot multiply two polynomials that contain non-numeric coefficients.')
        temp = Signomial.__mul__(self, other)
        temp = temp.as_polynomial()
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
        temp = temp.as_polynomial()
        return temp

    def __sub__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):
            raise RuntimeError('Cannot subtract a signomial from a polynomial (or vice versa).')
        temp = Signomial.__sub__(self, other)
        temp = temp.as_polynomial()
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
        temp = Signomial(self.alpha_c)
        temp = temp ** power
        temp = temp.as_polynomial()
        return temp

    def __call__(self, x):
        """
        Parameters
        ----------
        x : NumPy ndarray
            This vector can contain real numeric types and Polynomial objects.

        Returns
        -------
        val : float or Polynomial
            If ``x`` is purely numeric, then ``val`` is a float.
            If ``x`` contains Polynomial objects, then ``val`` is a Polynomial object.

        Examples
        --------
        Evaluating a Polynomial at a numeric point. ::

            p = Polynomial({(1,2,3): -1})
            x = np.array([3,2,1])
            print(p(x))  #  (3 ** 1) * (2 ** 2) * (1 ** 3) * (-1) == -12

        Evaluating a Polynomial on another polynomial. ::

            p = Polynomial({(2,): 1})  # represents lambda x: x ** 2
            z = Polynomial({(1,): 2, (0,): -1})  # represents lambda x: x - 1
            w = p(z)  # represents lambda x: (x - 1) ** 2
            print(w(1))  # equals zero
            print(w(2))  # equals one
            print(w(0))  # equals one

        Notes
        -----
        If ``x`` contains Polynomial entries, then those Polynomials must all be over the same
        number of variables. However, those Polynomials need not have the same number of variables
        as the current polynomial.

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
            other_v = other.query_coeff(np.array(k))
            if not cl.Expression.are_equivalent(other_v, v, rtol=1e-8):
                return False
        return True

    def remove_terms_with_zero_as_coefficient(self):
        """
        Update ``alpha``, ``c``, and ``alpha_c`` to remove nonconstant terms where ``c[i] == 0``.
        """
        Signomial.remove_terms_with_zero_as_coefficient(self)
        pass

    def query_coeff(self, a):
        """
        Returns the coefficient of the monomial ``lambda x: np.prod(np.power(a, x))`` for this Polynomial.
        """
        tup = tuple(a)
        if tup in self.alpha_c:
            return self.alpha_c[tup]
        else:
            return 0

    def constant_location(self):
        """
        Return the index ``i`` so that ``alpha[i, :]`` is the zero vector.
        """
        loc = Signomial.constant_location(self)
        return loc

    def even_locations(self):
        """
        Return the largest ndarray ``evens``, so that ``np.all(alpha[evens,:] % 2 == 0)``.
        """
        evens = np.where(~(self.alpha % 2).any(axis=1))[0]
        return evens

    def partial(self, i):
        """
        Compute the symbolic partial derivative of this polynomial, at coordinate ``i``.

        Parameters
        ----------
        i : int
            ``i`` must be an integer from 0 to ``self.n - 1``.

        Returns
        -------
        p : Polynomial
            The function obtained by differentiating this polynomial with respect to its i-th argument.
        """
        if i < 0 or i >= self.n:
            raise RuntimeError('This polynomial does not have an input at index ' + str(i) + '.')
        d = defaultdict(int)
        for j in range(self.m):
            if self.alpha[j, i] > 0:
                vec = self.alpha[j, :].copy()
                c = self.c[j] * vec[i]
                vec[i] -= 1
                d[tuple(vec.tolist())] = c
        d[self.n * (0,)] += 0
        p = Polynomial(d)
        return p

    def standard_multiplier(self):
        """
        Returns a Polynomial which is globally nonnegative by construction, for use as a modulator in SAGE hierarchies.
        The particular polynomial has exponents ``alpha = alpha[even_locations(), :]``, and a coefficient vector of
        all ones.
        """
        evens = self.even_locations()
        mult_alpha = self.alpha[evens, :].copy()
        mult_c = np.ones(len(evens))
        mult = Polynomial(mult_alpha, mult_c)
        if mult.alpha_c[self.n * (0,)] == 0:
            mult += 1
        return mult

    @property
    def sig_rep(self):
        """
        Return the signomial representative of the current Polynomial, as well as a list of constraints needed
        to enforce the relationship between the current Polynomial and the signomial representative.

        Returns
        -------
        sr : Signomial
            If this Signomial is globally nonnegative, then the current Polynomial is also globally nonnegative.

        sr_cons : list of coniclifts Constraints
            If the current Polynomial has nonconstant coefficients (i.e. some entries of ``c`` are coniclifts
            Variables), then ``sr`` will also have nonconstant coefficients. In order to enforce the relationship
            between ``sr`` and the current Polynomial, we may require constraints between ``c`` and ``sr.c``.
            Any such constraints are in this list.

        """
        # It is important that self._sig_rep.alpha == self.alpha.
        if self._sig_rep is None:
            self._compute_sig_rep()
        sr = self._sig_rep
        sr_cons = self._sig_rep_constrs
        return sr, sr_cons

    def _compute_sig_rep(self):
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

    def as_signomial(self):
        """
        Returns
        -------
        f : Signomial
            For every elementwise positive vector ``x``, we have ``self(x) == f(np.log(x))``.
        """
        f = Signomial(self.alpha_c)
        return f
