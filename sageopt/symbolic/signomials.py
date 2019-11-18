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
import warnings


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)

__EXPONENT_VECTOR_DECIMAL_POINTS__ = 7


def standard_sig_monomials(n):
    """
    Return ``y`` where ``y[i](x) = np.exp(x[i])`` for every numeric ``x`` of length ``n``.

    Parameters
    ----------
    n : int
        The signomials will be defined on :math:`R^n`.

    Returns
    -------
    y : ndarray
        An array  of length ``n``, containing Signomial objects.

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
    A symbolic representation for linear combinations of exponentials, composed with
    linear functions. The constructor for this class can be called in two different ways, and
    the arguments to the constructor have names which reflect the different possibilities.
    Refer to the Examples if you find the description of the constructor arguments unclear.

    Parameters
    ----------

    alpha_maybe_c : dict or ndarray

         If ``alpha_maybe_c`` is a dict, then it must be a dictionary from tuples-of-numbers to
         scalars. The keys will be converted to rows of a matrix which we call ``alpha``, and
         the values will be assembled into a vector which we call ``c``.

         If ``alpha_maybe_c`` is an ndarray, then the argument ``c`` must also be an ndarray,
         and ``c.size`` must equal the number of rows in ``alpha_maybe_c``.

    c : None or ndarray

        This value is only used when ``alpha_maybe_c`` is an ndarray. If that is the case, then
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

    The second way is to specify two arguments. In this case the first argument is an ndarray
    where the rows represent linear functionals, and the second argument is a vector of corresponding
    coefficients.::

        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        c = np.array([1, 2, 3])
        f = Signomial(alpha, c)
        x = np.random.randn(2)
        print(f(x) - c @ np.exp(alpha @ x))  # zero, up to rounding errors.

    Properties
    ----------

    n : int
        The dimension of the space over which this Signomial is defined. The number of columns in ``alpha``,
        and the length of tuples appearing in the dictionary ``alpha_c``.

    m : int
        The number of terms needed to express this Signomial in a basis of monomial functions
        ``lambda x: exp(a @ x)`` for row vectors ``a``. The signomial is presumed to include a constant term.

    alpha : ndarray
        Has shape ``(m, n)``. Each row specifies a vector appearing in an exponential function which
        defines this Signomial. The rows are ordered for consistency with the property ``c``.

    c : ndarray
        Has shape ``(m,)``. The scalar ``c[i]`` is this Signomial's coefficient on the basis function
        ``lambda x: exp(alpha[i, :] @ x)``. It is possible to have ``c.dtype == object``.

    alpha_c : defaultdict
        The keys of ``alpha_c`` are tuples of length ``n``, containing real numeric types (e.g int, float).
        These tuples define linear functions. This Signomial could be evaluated by the code snippet
        ``lambda x: np.sum([ alpha_c[a] * np.exp(a @ x) for a in alpha_c])``. The default value for this
        dictionary is zero.

    Attributes
    ----------

    metadata : dict
        A place for the user to store arbitrary information about this Signomial object.

    Notes
    -----

    Operator overloading.

        The operators ``+``, ``-``, and ``*`` are defined between pairs of Signomials, and pairs
        ``{s, t}`` where ``s`` is a Signomial and ``t`` is either numeric or a coniclifts Expression.

        A signomial ``s`` can be raised to a numeric power ``p`` by writing ``s**power``; if ``s.c``
        contains more than one nonzero entry, it can only be raised to nonnegative integer powers.

        The Signomial class also implements ``s1 / s2`` where ``s2`` is a numeric type or Signomial;
        if ``s2`` is a Signomial, then its coefficient vector ``s2.c`` can only contain one nonzero entry.

    Function evaluations.

        Signomial objects are callable. If ``s`` is a Signomial object and ``x`` is a numpy array of length ``s.n``,
        then ``s(x)`` computes the Signomial object as though it were any other elementary Python function.

        Signomial objects provide functions to compute gradients (equivalently, Jacobians) and Hessians.
        These methods operate by caching and evaluating symbolic representations of the relevant partial derivatives.

    Internal representations.

        Both ``self.alpha_c`` and ``(self.alpha, self.c)`` completely specify a Signomial object.
        You are free to use whichever is more convenient in a given context. Neither of these fields
        should be modified manually.
    """

    def __init__(self, alpha_maybe_c, c=None):
        # noinspection PyArgumentList
        if c is None:
            # noinspection PyArgumentList
            self._alpha_c = defaultdict(int, alpha_maybe_c)
        else:
            alpha = np.round(alpha_maybe_c, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
            alpha = alpha.tolist()
            if len(alpha) != c.size:  # pragma: no cover
                raise RuntimeError('alpha and c specify different numbers of terms')
            self._alpha_c = defaultdict(int)
            for j in range(c.size):
                self._alpha_c[tuple(alpha[j])] += c[j]
        self._n = len(list(self._alpha_c.items())[0][0])
        self._alpha_c[(0,) * self._n] += 0  # ensures that there's a constant term.
        self._m = len(self._alpha_c)
        self._alpha = None
        self._c = None
        self._grad = None
        self._hess = None
        self._arrays_stale = True
        self.metadata = dict()
        pass

    def _cache_grad(self):
        if self._grad is None:
            g = np.empty(shape=(self._n,), dtype=object)
            for i in range(self._n):
                g[i] = self.partial(i)
            self._grad = g
        pass

    def _cache_hess(self):
        if self._hess is None:
            H = np.empty(shape=(self._n, self._n), dtype=object)
            for i in range(self._n):
                ith_partial = self.partial(i)
                for j in range(i+1):
                    curr_partial = ith_partial.partial(j)
                    H[i, j] = curr_partial
                    H[j, i] = curr_partial
            self._hess = H
        pass

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def alpha(self):
        if self._arrays_stale:
            self._update_alpha_c_arrays()
        return self._alpha

    @property
    def c(self):
        if self._arrays_stale:
            self._update_alpha_c_arrays()
        return self._c

    @property
    def alpha_c(self):
        return self._alpha_c

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
        if tup in self._alpha_c:
            return self._alpha_c[tup]
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
        for k, v in self._alpha_c.items():
            alpha.append(k)
            c.append(v)
        self._alpha = np.array(alpha)
        self._c = np.array(c)
        if self._c.dtype == np.dtype('O'):
            self._c = cl.Expression(self._c)
        self._arrays_stale = False

    def __add__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self._n
            d = {tup: other}
            other = Signomial(d)
        if not other.n == self._n:
            raise RuntimeError('Cannot add Signomials with different numbers of variables.')
        # noinspection PyArgumentList
        d = defaultdict(int, self._alpha_c)
        for k, v in other._alpha_c.items():
            d[k] += v
        res = Signomial(d)
        res.remove_terms_with_zero_as_coefficient()
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self._n
            d = {tup: other}
            other = Signomial(d)
        if not other.n == self._n:
            raise RuntimeError('Cannot multiply Signomials with different numbers of variables.')
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
        if type(power) not in __NUMERIC_TYPES__:
            raise RuntimeError('Cannot raise a signomial to non-numeric powers.')
        if self.c.dtype not in __NUMERIC_TYPES__:
            if isinstance(self.c, cl.Expression) and not self.c.is_constant():
                raise RuntimeError('Cannot exponentiate signomials with symbolic coefficients.')
        if power % 1 == 0 and power >= 0:
            power = int(power)
            if power == 0:
                # noinspection PyTypeChecker
                return Signomial({(0,) * self._n: 1})
            else:
                s = Signomial(self._alpha_c)
                for t in range(power - 1):
                    s = s * self
                return s
        else:
            d = dict((k, v) for (k, v) in self._alpha_c.items() if v != 0)
            if len(d) != 1:
                raise ValueError('Only signomials with exactly one term can be raised to power %s.', power)
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

        :param x: either a scalar (if self._n == 1), or a numpy _n-d array with len(x.shape) <= 2
        and x.shape[0] == self._n.
        :return:  If x is a scalar or an _n-d array of shape (self._n,), then "val" is a numeric
        type equal to the signomial evaluated at x. If instead x is of shape (self._n, k) for
        some positive integer k, then "val" is a numpy _n-d array of shape (k,), with val[i]
        equal to the current signomial evaluated on the i^th column of x.

        This function's behavior is undefined when x is not a scalar and has len(x.shape) > 2.
        """
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1 and x.ndim == 0):
            x = np.array([np.asscalar(np.array(x))])  # coerce into a 1d array of shape (1,).
        if not x.shape[0] == self._n:
            msg = 'Domain is R^' + str(self._n) + 'but x is in R^' + str(x.shape[0])
            raise ValueError(msg)
        if x.ndim > 2:
            msg = 'Signomials cannot be called on ndarrays with more than 2 dimensions.'
            raise ValueError(msg)
        x = x.astype(np.longdouble)
        exponents = np.dot(self.alpha.astype(np.longdouble), x)
        linear_vars = np.exp(exponents).astype(np.longdouble)
        val = np.dot(self.c, linear_vars)
        return val

    def __hash__(self):
        return hash(frozenset(self._alpha_c.items()))

    def __eq__(self, other):
        if not isinstance(other, Signomial):
            return False
        if self._m != other._m:
            return False
        for k in self._alpha_c:
            v = self._alpha_c[k]
            other_v = other.query_coeff(np.array(k))
            if not cl.Expression.are_equivalent(other_v, v, rtol=1e-8):
                return False
        return True

    def remove_terms_with_zero_as_coefficient(self):
        """
        Update ``alpha``, ``c``, and ``alpha_c`` to remove nonconstant terms where ``c[i] == 0``.
        """
        d = dict()
        for (k, v) in self._alpha_c.items():
            if (not isinstance(v, __NUMERIC_TYPES__)) or v != 0:
                d[k] = v
        # noinspection PyArgumentList
        self._alpha_c = defaultdict(int, d)
        tup = (0,) * self._n
        self._alpha_c[tup] += 0
        self._m = len(self._alpha_c)
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
        if i < 0 or i >= self._n:  # pragma: no cover
            raise RuntimeError('This Signomial does not have an input at index ' + str(i) + '.')
        d = defaultdict(int)
        for j in range(self._m):
            c = self.c[j] * self.alpha[j, i]
            if (not isinstance(c, __NUMERIC_TYPES__)) or c != 0:
                vec = self.alpha[j, :].copy()
                d[tuple(vec.tolist())] += c
        d[self._n * (0,)] += 0
        p = Signomial(d)
        return p

    def grad_val(self, x):
        """
        Return the gradient of this Signomial (as an ndarray) at the point ``x``.
        """
        _ = self.grad  # construct the function handles.
        g = np.zeros(self._n)
        for i in range(self._n):
            g[i] = self._grad[i](x)
        return g

    def hess_val(self, x):
        """
        Return the Hessian of this Signomial (as an ndarray) at the point ``x``.
        """
        if self._hess is None:
            _ = self.hess  # ignore the return value
        H = np.zeros(shape=(self._n, self._n))
        for i in range(self._n):
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
            For every vector ``x``, we have ``self(x) == f(np.exp(x))``.
        """
        from sageopt.symbolic.polynomials import Polynomial
        f = Polynomial(self.alpha, self.c)
        return f


class SigDomain(object):
    """
    Represent a convex set :math:`X \\subset R^n`, for use in signomial conditional SAGE relaxations.

    Parameters
    ----------
    n : int
        The dimension of the space in which this set lives.

    Example
    -------
    Suppose you want to represent the :math:`\\ell_2` unit ball in :math:`R^{5}`.
    This can be done as follows, ::

        import sageopt as so
        import sageopt.coniclifts as cl
        x_var = cl.Variable(shape=(5,), name='temp')
        cons = [cl.vector2norm(x_var) <= 1]
        X = so.SigDomain(5)
        X.parse_coniclifts_constraints(cons)

    As written, that SigDomain cannot be used for solution recovery from SAGE relaxations.
    To fully specify a SigDomain, you need to set attributes ``gts`` and ``eqs``, which
    are lists of inequality constraints (``g(x) >= 0``) and equality constraints
    (``g(x) == 0``) respectively. The following code completes the example above ::

        import numpy as np
        my_gts = [lambda dummy_x: 1 - np.linalg.norm(dummy_x, ord=2)]
        X.gts = my_gts
        X.eqs = []

    This class does not check for correctness of ``eqs`` and ``gts``. It is up to the user
    to ensure these values represent this SigDomain in the intended way.

    Notes
    -----
    The constructor accepts the following keyword arguments:

    coniclifts_cons: list of coniclifts.constraints.Constraint
        Constraints over a single coniclifts Variable, which define this SigDomain.

    gts : list of callable
        Inequality constraint functions (``g(x) >= 0``) which can be used to represent ``X``.

    eqs : list of callable
        Equality constraint functions (``g(x) == 0``) which can be used to represent ``X``.

    check_feas : bool
        Whether or not to check that ``X`` is nonempty. Defaults to True.

    AbK : tuple
        Specify a convex set in the coniclifts standard. ``AbK[0]`` is a SciPy sparse
        matrix. The first ``_n`` columns of this matrix correspond to the variables over
        which this set is supposed to be defined. Any remaining columns are for auxiliary
        variables.

    Only one of ``AbK`` and ``coniclifts_cons`` can be provided upon construction.
    If more than one of these value is provided, the constructor will raise an error.
    """

    __VALID_KWARGS__ = {'gts', 'eqs', 'AbK', 'coniclifts_cons', 'check_feas'}

    def __init__(self, n, **kwargs):
        for kw in kwargs:
            if kw not in SigDomain.__VALID_KWARGS__:  # pragma: no cover
                msg = 'Provided keyword argument "' + kw + '" is not in the list'
                msg += ' of allowed keyword arguments: \n'
                msg += '\t ' + str(SigDomain.__VALID_KWARGS__)
        self.n = n
        self.A = None
        self.b = None
        self.K = None
        self.gts = kwargs['gts'] if 'gts' in kwargs else None
        self.eqs = kwargs['eqs'] if 'eqs' in kwargs else None
        self.check_feas = kwargs['check_feas'] if 'check_feas' in kwargs else True
        self._constraints = None  # optional
        self._x = None  # optional
        self._variables = None  # optional
        if 'AbK' in kwargs:
            self.A, self.b, self.K = kwargs['AbK']
            if self.check_feas:
                self._check_feasibility()
        if 'coniclifts_cons' in kwargs:
            if self.A is not None:  # pragma: no cover
                msg = 'Keyword arguments "AbK" and "coniclifts_cons" are mutually exclusive.'
                raise RuntimeError(msg)
            else:
                self.parse_coniclifts_constraints(kwargs['coniclifts_cons'])
        pass

    def _check_feasibility(self):
        A, b, K = self.A, self.b, self.K
        temp_x = cl.Variable(shape=(A.shape[1],), name='temp_x')
        cons = [cl.PrimalProductCone(A @ temp_x + b, K)]
        prob = cl.Problem(cl.MIN, cl.Expression([0]), cons)
        prob.solve(verbose=False, solver='ECOS')
        if not prob.value < 1e-7:
            if prob.value is np.NaN:  # pragma: no cover
                msg = 'SigDomain constraints could not be verified as feasible.'
                msg += '\n Proceed with caution!'
                warnings.warn(msg)
            else:
                msg1 = 'SigDomain constraints seem to be infeasible.\n'
                msg2 = 'Feasibility problem\'s status: ' + prob.status + '\n'
                msg3 = 'Feasibility problem\'s  value: ' + str(prob.value) + '\n'
                msg4 = 'The objective was "minimize 0"; we expect problem value < 1e-7. \n'
                msg = msg1 + msg2 + msg3 + msg4
                raise RuntimeError(msg)
        pass

    def parse_coniclifts_constraints(self, constraints):
        """
        Modify this SigDomain object, so that it represents the set of values
        which satisfy the provided constraints.

        Parameters
        ----------
        constraints : list of coniclifts.Constraint
            The provided constraints must be defined over a single coniclifts Variable.

        """
        variables = cl.compilers.find_variables_from_constraints(constraints)
        if len(variables) != 1:
            raise RuntimeError('The system of constraints must be defined over a single Variable object.')
        self._constraints = constraints
        self._x = variables[0]
        if self._x.size != self.n:
            msg = 'The provided constraints are over a variable of dimension '
            msg += str(self._x.size) + ', but this SigDomain was declared as dimension ' + str(self.n) + '.'
            raise RuntimeError(msg)
        A, b, K, variable_map, all_variables, _ = cl.compile_constrained_system(constraints)
        A = A.toarray()
        selector = variable_map[self._x.name].ravel()
        A0 = np.hstack((A, np.zeros(shape=(A.shape[0], 1))))
        A_lift = A0[:, selector]
        aux_len = A.shape[1] - np.count_nonzero(selector != -1)
        if aux_len > 0:
            A_aux = A[:, -aux_len:]
            A_lift = np.hstack((A_lift, A_aux))
        self.A = A_lift
        self.b = b
        self.K = K
        if self.check_feas:
            self._check_feasibility()
        pass

    def check_membership(self, x_val, tol):
        """
        Evaluate ``self.gts`` and ``self.eqs`` at ``x_val``,
        to check if ``x_val`` belongs to this SigDomain.

        Parameters
        ----------
        x_val : ndarray
            Check if ``x_val`` belongs in this domain.
        tol : float
            Infeasibility tolerance.

        Returns
        -------
        res : bool
            True iff ``x_val`` belongs in the domain represented by ``self``, up
            to infeasibility tolerance ``tol``.

        """
        if any([g(x_val) < -tol for g in self.gts]):
            return False
        if any([abs(g(x_val)) > tol for g in self.eqs]):
            return False
        return True
