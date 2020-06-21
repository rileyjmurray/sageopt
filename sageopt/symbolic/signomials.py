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
from sageopt.symbolic import utilities as sym_util
import numpy as np
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
    A concrete representation for a function of the form
    :math:`x \\mapsto \\sum_{i=1}^m c_i \\exp(\\alpha_i \cdot x)`.

    Operator overloading.

        The operators ``+``, ``-``, and ``*`` are defined between pairs of Signomials, and pairs
        ``{s, t}`` where ``s`` is a Signomial and ``t`` is "scalar-like." Specific examples of
        "scalar-like" objects include numeric types, and coniclifts Expressions of size one.

        A Signomial ``s`` can be raised to a numeric power ``p`` by writing ``s**p``; if ``s.c``
        contains more than one nonzero entry, it can only be raised to nonnegative integer powers.

        The Signomial class implements ``s1 / s2`` where ``s2`` is a numeric type or Signomial;
        if ``s2`` is a Signomial, then its coefficient vector ``s2.c`` can only contain one nonzero entry.

    Function evaluations.

        Signomial objects are callable. If ``s`` is a Signomial object and ``x`` is a numpy array of length ``s.n``,
        then ``s(x)`` computes the Signomial object as though it were any other elementary Python function.

        Signomial objects provide functions to compute gradients (equivalently, Jacobians) and Hessians.
        These methods operate by caching and evaluating symbolic representations of partial derivatives.

    Parameters
    ----------

    alpha : ndarray

         The rows of ``alpha`` comprise this Signomial's exponent vectors.

    c : None or ndarray

        An ndarray of coefficients, with one coefficient for each row in alpha.

    Examples
    --------

    There are three ways to make Signomial objects. The first way is to call the constructor::

        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        c = np.array([1, 2, 3])
        f = Signomial(alpha, c)
        x = np.random.randn(2)
        print(f(x) - c @ np.exp(alpha @ x))  # zero, up to rounding errors.

    You can also use ``Signomial.from_dict`` which maps exponent vectors (represented as
    tuples) to scalars::

        alpha_and_c = {(1,): 2}
        f = Signomial.from_dict(alpha_and_c)
        print(f(1))  # equal to 2 * np.exp(1),
        print(f(0))  # equal to 2.

    The final way to construct a Signomial is with algebraic syntax, like::

        y = sageopt.standard_sig_monomials(2)  # y[i] represents exp(x[i])
        f = (y[0] - y[1]) ** 3 + 1 / y[0]  # a Signomial in two variables
        x = np.array([1, 1])
        print(f(x))  # np.exp(-1), up rounding errors.

    Signomial objects are not limited to numeric problem data for ``alpha`` and ``c``.
    In fact, it's very common to have ``c`` contain a coniclifts Expression. For example,
    if we started with a Signomial ``f`` and then updated ::

        gamma = sageopt.coniclifts.Variable()
        f =  f - gamma

    then ``f.c`` would be a coniclifts Expression depending on the variable ``gamma``.

    Notes
    -----

    Signomial objects have a dictionary attribute called ``metadata``. You can store any information
    you'd like in this dictionary. However, the information in this dictionary will not automatically be
    propogated when creating new Signomial objects (as happens when performing arithmetic on Signomials).
    """

    def __init__(self, alpha, c):
        if isinstance(alpha, list):
            alpha = np.ndarray(alpha)
        if alpha.shape[0] != c.size:  # pragma: no cover
            raise ValueError('alpha and c specify different numbers of terms')
        if isinstance(c, np.ndarray) and not isinstance(c, cl.Expression) and c.dtype == object:
            raise ValueError('If c is an ordinary numpy array, it cannot have dtype == object.')
        if isinstance(alpha, np.ndarray):
            alpha = np.round(alpha, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
            alpha, c = sym_util.consolidate_basis_funcs(alpha, c)
        self._alpha = alpha
        self._c = c
        self._n = alpha.shape[1]
        self._m = alpha.shape[0]
        self._alpha_c = None
        self._grad = None
        self._hess = None
        self.metadata = dict()
        pass

    def _cache_grad(self):
        if self._grad is None:
            g = np.empty(shape=(self._n,), dtype=object)
            for i in range(self._n):
                g[i] = self._partial(i)
            self._grad = g
        pass

    def _cache_hess(self):
        if self._hess is None:
            H = np.empty(shape=(self._n, self._n), dtype=object)
            for i in range(self._n):
                ith_partial = self._partial(i)
                for j in range(i+1):
                    curr_partial = ith_partial._partial(j)
                    H[i, j] = curr_partial
                    H[j, i] = curr_partial
            self._hess = H
        pass

    @property
    def n(self):
        """
        The dimension of the space over which this Signomial is defined;
        this Signomial accepts inputs in :math:`\\mathbb{R}^{n}`.
        """
        return self._n

    @property
    def m(self):
        """
        The number of monomial basis functions :math:`x \\mapsto \\exp(a \\cdot x)`
        used by this Signomial.
        """
        return self._m

    @property
    def alpha(self):
        """
        Has shape ``(m, n)``. Each row specifies a vector appearing in an exponential function which
        defines this Signomial. The rows are ordered for consistency with the property ``c``.
        """
        return self._alpha

    @property
    def c(self):
        """
        Has shape ``(m,)``. The scalar ``c[i]`` is this Signomial's coefficient on the basis function
        ``lambda x: exp(alpha[i, :] @ x)``. It's possible to have ``c.dtype == object``.
        """
        return self._c

    @property
    def alpha_c(self):
        """
        The keys of ``alpha_c`` are tuples of length ``n``, containing real numeric types (e.g int, float).
        These tuples define linear functions. This Signomial could be evaluated by the code snippet
        ``lambda x: np.sum([ alpha_c[a] * np.exp(a @ x) for a in alpha_c])``.
        """
        if self._alpha_c is None:
            self._build_alpha_c_dict()
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
        if tup in self.alpha_c:
            return self.alpha_c[tup]
        else:
            return 0

    def constant_location(self):
        """
        Return the index ``i`` so that ``alpha[i, :]`` is the zero vector.
        """
        if not isinstance(self.alpha, np.ndarray):
            raise NotImplementedError()
        res = np.where((self.alpha == 0).all(axis=1))[0]
        if res.size == 0:
            return None
        else:
            return res[0]

    def _build_alpha_c_dict(self):
        alpha, c = self._alpha, self._c
        if not isinstance(alpha, np.ndarray):
            raise NotImplementedError()
        self._alpha_c = dict()
        for j in range(c.size):
            self._alpha_c[tuple(alpha[j, :])] = c[j]

    def __add__(self, other):
        try:
            other = self.upcast_to_signomial(other)
        except ValueError:
            return other.__radd__(self)
        if not other.n == self._n:
            raise RuntimeError('Cannot add Signomials with different numbers of variables.')
        res = Signomial.sum([self, other])
        res = res.without_zeros()
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            other = self.upcast_to_signomial(other)
        except ValueError:
            return other.__rmul__(self)
        if not other.n == self._n:
            raise RuntimeError('Cannot multiply Signomials with different numbers of variables.')
        res = Signomial.product(self, other)
        res = res.without_zeros()
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
            msg = 'Cannot exponentiate signomials with symbolic coefficients.'
            if not isinstance(self.c, cl.Expression):
                raise RuntimeError(msg)
            elif not self.c.is_constant():
                raise RuntimeError(msg)
        if power % 1 == 0 and power >= 0:
            power = int(power)
            if power == 0:
                # noinspection PyTypeChecker
                return self.upcast_to_signomial(1)
            else:
                #TODO: implement a faster version of this in utilties.py,
                # by using multinomial coefficients. Once this is done,
                # add new tests for correctness of ``__pow__``, since it would
                # no longer follow from correctness of ``__mul__``.
                s = Signomial(self.alpha, self.c)
                for t in range(power - 1):
                    s = s * self
                return s
        else:
            d = dict((k, v) for (k, v) in self.alpha_c.items() if v != 0)
            if len(d) != 1:
                raise ValueError('Only signomials with exactly one term can be raised to power %s.', power)
            v = list(d.values())[0]
            if v < 0 and not power % 1 == 0:
                raise ValueError('Cannot compute non-integer power %s of coefficient %s', power, v)
            alpha_tup = tuple(power * ai for ai in list(d.keys())[0])
            c = float(v) ** power
            s = Signomial.from_dict({alpha_tup: c})
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
        if not isinstance(self.alpha, np.ndarray):
            # assuming cvxpy Expression, could substitute alpha = alpha.value
            raise NotImplementedError()
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
        return hash(frozenset(self.alpha_c.items()))

    def __eq__(self, other):
        if not isinstance(other, Signomial):
            return False
        if self._m != other._m:
            return False
        if self.c.dtype not in __NUMERIC_TYPES__ or other.c.dtype not in __NUMERIC_TYPES__:
            return False
        if not isinstance(self._alpha, np.ndarray) or not isinstance(other.alpha, np.ndarray):
            return False
        for k in self.alpha_c:
            v = self.alpha_c[k]
            other_v = other.query_coeff(np.array(k))
            if abs(v - other_v) > 1e-8:
                return False
        return True

    def without_zeros(self):
        """
        Return a Signomial which is symbolically equivalent to ``self``,
        but which doesn't track basis functions ``alpha[i,:]`` for which ``c[i] == 0``.
        """
        if self.m == 1:
            return self
        if isinstance(self.c, cl.Variable):
            return self
        to_drop = sym_util.find_zero_entries(self.c)
        if len(to_drop) == 0:
            return self
        elif len(to_drop) == self.m:
            # noinspection PyTypeChecker
            return self.upcast_to_signomial(0)
        else:
            keepers = np.ones(self.m, dtype=bool)
            keepers[to_drop] = False
            c = self.c[keepers]
            alpha = self.alpha[keepers, :]
            s = Signomial(alpha, c)
            return s

    def _partial(self, i):
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
        d = dict()
        for j in range(self._m):
            vec = self.alpha[j, :]
            c = self.c[j] * vec[i]
            if (not isinstance(c, __NUMERIC_TYPES__)) or c != 0:
                d[tuple(vec)] = c
        if len(d) == 0:
            d[(0,) * self._n] = 0
        p = Signomial.from_dict(d)
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

    def upcast_to_signomial(self, other):
        if isinstance(other, Signomial):
            return other
        alpha = np.zeros(shape=(1, self.n))
        if isinstance(other, __NUMERIC_TYPES__):
            s = Signomial(alpha, np.array([other]))
            return s
        elif isinstance(other, cl.base.ScalarExpression):
            s = Signomial(alpha, cl.Expression([other]))
            return s
        elif not hasattr(other, 'size'):
            raise ValueError()
        elif other.size > 1:
            raise ValueError()
        else:
            s = Signomial(alpha, other.flatten())
            return s

    @staticmethod
    def product(f1, f2):
        alpha1, c1 = f1.alpha, f1.c
        if isinstance(alpha1, np.ndarray):
            alpha1 = alpha1.astype(np.float_)
        m1 = alpha1.shape[0]
        alpha2, c2 = f2.alpha, f2.c
        if isinstance(alpha2, np.ndarray):
            alpha2 = alpha2.astype(np.float_)
        m2 = alpha2.shape[0]
        # lift alpha1, c1 into the product basis
        tile_idxs = np.tile(np.arange(m1), reps=m2)
        alpha1_lift = alpha1[tile_idxs, :]
        c1_lift = c1[tile_idxs]
        # lift alpha2, c2 into the product basis
        repeat_idxs = np.repeat(np.arange(m2), repeats=m1)
        alpha2_lift = alpha2[repeat_idxs, :]
        c2_lift = c2[repeat_idxs]
        # explicitly form the product basis, and the coefficient vector for the product
        alpha_lift = alpha1_lift + alpha2_lift
        if isinstance(alpha_lift, np.ndarray):
            alpha_lift = np.round(alpha_lift,
                                  decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
        c_lift = c1_lift * c2_lift
        p = type(f1)(alpha_lift, c_lift)
        return p

    @staticmethod
    def sum(funcs):
        if len(funcs) == 0:
            raise ValueError()
        elif any(not isinstance(f, Signomial) for f in funcs):
            raise ValueError()
        elif len(funcs) == 1:
            return funcs[0]
        mats = [f.alpha for f in funcs]
        alpha, all_crs = sym_util.align_basis_matrices(mats)
        num_rows = alpha.shape[0]
        coeff_vecs = [f.c for f in funcs]
        lifted_cs = sym_util.lift_basis_coeffs(coeff_vecs, all_crs, num_rows)
        c = sum(lifted_cs)
        s = type(funcs[0])(alpha, c)
        return s

    @staticmethod
    def from_dict(d):
        """
        Construct a Signomial object which represents the function::

            lambda x: np.sum([ d[a] * np.exp(a @ x) for a in d])

        Parameters
        ----------
        d : Dict[Tuple[Float], Float]

        Returns
        -------
        s : Signomial
        """
        alpha = []
        c = []
        for k, v in d.items():
            alpha.append(k)
            c.append(v)
        alpha = np.array(alpha)
        c = np.array(c)
        s = Signomial(alpha, c)
        s._alpha_c = d
        return s


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
        matrix. The first ``n`` columns of this matrix correspond to the variables over
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
        self._lift_x = None  # for computing support function
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
        if self._lift_x is None:
            self._lift_x = cl.Variable(shape=(A.shape[1],), name='temp_x')
        cons = [cl.PrimalProductCone(A @ self._lift_x + b, K)]
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

    def suppfunc(self, y):
        """
        The support function of the convex set :math:`X` associated with this SigDomain,
        evaluated at :math:`y`:

        .. math::

            \\sigma_X(y) \\doteq \\max\\{ y^\\intercal x \\,:\\, x \\in X \\}.
        """
        if isinstance(y, cl.Expression):
            y = y.value
        if self._lift_x is None:
            self._lift_x = cl.Variable(self.A.shape[1])
        objective = y @ self._lift_x
        cons = [cl.PrimalProductCone(self.A @ self._lift_x + self.b, self.K)]
        prob = cl.Problem(cl.MAX, objective, cons)
        prob.solve(solver='ECOS', verbose=False)
        if prob.status == cl.FAILED:
            return np.inf
        else:
            return prob.value
