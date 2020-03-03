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
from sageopt.symbolic.signomials import Signomial
import warnings


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def standard_poly_monomials(n):
    """
    Return ``x`` where ``x[i](z) = z[i]`` for all numeric ``z`` of length ``n``.

    Parameters
    ----------
    n : int
        The polynomials will be defined on :math:`R^n`.

    Returns
    -------
    x : ndarray
        An array  of length ``n``, containing Polynomial objects.

    Example
    -------
    This function is useful for constructing Polynomials using algebraic syntax. ::

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
    A concrete representation for a multivariate polynomial
    :math:`x \\mapsto \\sum_{i=1}^m c_i \\prod_{j=1}^n x_j^{\\alpha_{ij}}`.

    Polynomial objects provide the same operator overloading as the Signomial class.
    In particular, Polynomials implement ``+``, ``-``, ``*``, ``/``, and ``**``.

    Parameters
    ----------

    alpha : ndarray

        The returned Polynomial will have one monomial for each row in alpha. The number
        ``alpha[i, j]`` represents the power of variable ``j`` on the i-th monomial.

    c : ndarray

        The number ``c[i]`` os the coefficient on the i-th monomial, where the i-th
        monomial is defined by ``alpha[i, :]``.

    Examples
    --------

    There are three ways to make Polynomial objects. The first way is to call the constructor::

        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        c = np.array([1, 2, 3])
        f = Polynomial(alpha, c)
        x = np.array([-4, 7])
        val = f(x)  # val = 1 * (-4) + 2 * (7) + 3 * (-4 * 7)
        print(val)  # -74

    You can also use ``Polynomial.from_dict``, which maps exponent vectors (represented as
    tuples) to scalars::

        alpha_and_c = {(1,): 2}
        f = Polynomial.from_dict(alpha_and_c)
        print(f(1))  # equal to 2.
        print(f(-3))  # equal to -6.

    The final way to construct a Polynomial is with algebraic syntax, like::

        x = standard_poly_monomials(3)
        f = (x[0] ** 2) * x[2] + 4 * x[1] * x[2] * x[0]
        z = np.array([1, 2, 3])
        val = f(z)  # val = (1**2) * 3 + 4 * (2 * 3 * 1)
        print(val)  # 27.

    Polynomial objects are not limited to numeric problem data for ``alpha`` and ``c``.
    In fact, it's very common to have ``c`` contain a coniclifts Expression. For example,
    if we started with a Polynomial ``f`` and then updated ::

        gamma = sageopt.coniclifts.Variable()
        f =  f - gamma

    then ``f.c`` would be a coniclifts Expression depending on the variable ``gamma``.

    Notes
    -----
    The Polynomial class subclasses the Signomial class. This is done because most algebraic operations between
    polynomials and signomials are identical. However it is important to remember that polynomials and signomials
    evaluate in very different ways.

    Polynomial objects have a dictionary attribute called ``metadata``. You can store any information
    you'd like in this dictionary. However, the information in this dictionary will not automatically be
    propogated when creating new Polynomial objects (as happens when performing arithmetic on Polynomials).
    """

    def __init__(self, alpha, c):
        Signomial.__init__(self, alpha, c)
        if not np.all(self.alpha % 1 == 0):  # pragma: no cover
            raise RuntimeError('Exponents must belong the the integer lattice.')
        if not np.all(self.alpha >= 0):  # pragma: no cover
            raise RuntimeError('Exponents must be nonnegative.')
        self._sig_rep = None
        self._sig_rep_constrs = []
        pass

    @property
    def n(self):
        """
        The dimension of the space over which this Polynomial is defined;
        this Polynomial accepts inputs in :math:`\\mathbb{R}^{n}`.
        """
        return self._n

    @property
    def m(self):
        """
        The number of monomial basis functions :math:`x \\mapsto \\prod_{j=1}^n x_j^{a_j}`
        used by this Polynomial.
        """
        return self._m

    @property
    def alpha(self):
        """
        Has shape ``(m, n)``. Each row specifies the exponents of a monomial which
        appear in this Polynomial. The rows are ordered for consistency with the property ``c``.
        """
        return self._alpha

    @property
    def c(self):
        """
        Has shape ``(m,)``. The scalar ``c[i]`` is this Polynomial's coefficient on the
        basis function ``lambda x: np.prod(np.power(x, alpha[i, :]))``.
        """
        return self._c

    @property
    def alpha_c(self):
        """
        The keys of ``alpha_c`` are tuples of length ``n``, containing real numeric types (e.g int, float).
        These tuples define monomial exponents. This Polynomial could be evaluated by the code snippet
        ``lambda x: np.sum([ alpha_c[a] * np.prod(np.power(x,a)) for a in alpha_c])``.
        """
        if self._alpha_c is None:
            self._build_alpha_c_dict()
        return self._alpha_c

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

    @property
    def grad(self):
        """
        A numpy ndarray of shape ``(n,)`` whose entries are Polynomials. For a numpy ndarray ``x``,
        ``grad[i](x)`` is the partial derivative of this Polynomial with respect to coordinate ``i``,
        evaluated at ``x``. This array is constructed only when necessary, and is cached upon construction.
        """
        Signomial._cache_grad(self)  # This ends up calling the Polynomial "_partial" function.
        return self._grad

    @property
    def hess(self):
        """
        A numpy ndarray of shape ``(n, n)``, whose entries are Polynomials. For a numpy ndarray ``x``,
        ``hess[i,j](x)`` is the (i,j)-th partial derivative of this Polynomial, evaluated at ``x``.
        This array is constructed only when necessary, and is cached upon construction.
        """
        Signomial._cache_hess(self)  # This ends up calling the Polynomial "_partial" function.
        return self._hess

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            if isinstance(other, Signomial):  # pragma: no cover
                raise RuntimeError('Cannot multiply signomials and polynomials.')
            other = self.upcast_to_signomial(other)
            other = other.as_polynomial()
        self_var_coeffs = (self.c.dtype not in __NUMERIC_TYPES__)
        other_var_coeffs = (other.c.dtype not in __NUMERIC_TYPES__)
        if self_var_coeffs and other_var_coeffs:  # pragma: no cover
            raise RuntimeError('Cannot multiply two polynomials that contain non-numeric coefficients.')
        temp = Signomial.__mul__(self, other)
        temp = temp.as_polynomial()
        return temp

    def __truediv__(self, other):
        if not isinstance(other, __NUMERIC_TYPES__):  # pragma: no cover
            raise RuntimeError('Cannot divide a polynomial by the non-numeric type: ' + type(other) + '.')
        other_inv = 1 / other
        return self.__mul__(other_inv)

    def __add__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):  # pragma: no cover
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
        temp = Signomial(self.alpha, self.c)
        temp = temp ** power
        temp = temp.as_polynomial()
        return temp

    def __call__(self, x):
        """
        Parameters
        ----------
        x : ndarray
            This vector can contain real numeric types and Polynomial objects.

        Returns
        -------
        val : float or Polynomial
            If ``x`` is purely numeric, then ``val`` is a float.
            If ``x`` contains Polynomial objects, then ``val`` is a Polynomial object.

        Examples
        --------
        Evaluating a Polynomial at a numeric point. ::

            p = Polynomial.from_dict({(1,2,3): -1})
            x = np.array([3,2,1])
            print(p(x))  #  (3 ** 1) * (2 ** 2) * (1 ** 3) * (-1) == -12

        Evaluating a Polynomial on another polynomial. ::

            p = Polynomial.from_dict({(2,): 1})  # represents lambda x: x ** 2
            z = Polynomial.from_dict({(1,): 2, (0,): -1})  # represents lambda x: 2*x - 1
            w = p(z)  # represents lambda x: (2*x - 1) ** 2
            print(w(0.5))  # equals zero
            print(w(1))  # equals one
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
        else:
            res = Signomial.__eq__(self, other)
            return res

    def without_zeros(self):
        """
        Return a Polynomial which is symbolically equivalent to ``self``,
        but which doesn't track basis functions ``alpha[i,:]`` for which ``c[i] == 0``.
        """
        p = Signomial.without_zeros(self)
        p = p.as_polynomial()
        return p

    def query_coeff(self, a):
        """
        Returns the coefficient of the monomial ``lambda x: np.prod(np.power(x, a))`` for this Polynomial.
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

    def _partial(self, i):
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
        d = dict()
        for j in range(self.m):
            vec = self.alpha[j, :].copy()
            if vec[i] > 0:
                c = self.c[j] * vec[i]
                vec[i] -= 1
                tup = tuple(vec)
                if tup not in d:
                    d[tup] = c
                else:
                    d[tup] += c
        if len(d) == 0:
            d[(0,) * self.n] = 0
        p = Polynomial.from_dict(d)
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
        return mult

    def _compute_sig_rep(self):
        self._sig_rep = None
        self._sig_rep_constrs = []
        sigrep_c = np.zeros(shape=(self.m,), dtype=object)
        need_vars = []
        for i, row in enumerate(self.alpha):
            if np.any(row % 2 != 0):
                if isinstance(self.c[i], __NUMERIC_TYPES__):
                    sigrep_c[i] = -abs(self.c[i])
                elif self.c[i].is_constant():
                    sigrep_c[i] = -abs(self.c[i].value)
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
        if sigrep_c.dtype == object:
            sigrep_c = cl.Expression(sigrep_c)
        self._sig_rep = Signomial(self.alpha, sigrep_c)
        pass

    def as_signomial(self):
        """
        Returns
        -------
        f : Signomial
            For every elementwise positive vector ``x``, we have ``self(x) == f(np.log(x))``.
        """
        f = Signomial(self.alpha, self.c)
        return f

    def upcast_to_polynomial(self, other):
        s = Signomial.upcast_to_signomial(self, other)
        p = s.as_polynomial()
        return p

    @staticmethod
    def from_dict(d):
        """
        Construct a Polynomial object which represents the function::

            lambda x: np.sum([ d[a] * np.prod(np.power(x, a)) for a in d]).

        Parameters
        ----------
        d : Dict[Tuple[Float], Float]

        Returns
        -------
        s : Polynomial
        """
        s = Signomial.from_dict(d)
        p = s.as_polynomial()
        p._alpha_c = s.alpha_c
        return p


class PolyDomain(object):
    """
    Represent a set :math:`X \\subset R^n` satisfying the following properties:

     1. invariance under reflection about the :math:`n`
        hyperplanes :math:`H_i = \\{ (x_1,\\ldots,x_n) : x_i = 0 \\}`.

     2. the set :math:`X \\cap R^n_{++}` is is log convex, and

     3. the closure of :math:`X \\cap R^n_{++}` equals :math:`X \\cap R^n_+`

    Such sets are used in polynomial conditional SAGE relaxations.

    Parameters
    ----------
    n : int
        The dimension of the space in which this set lives.

    Other Parameters
    ----------------

    logspace_cons: list of coniclifts.constraints.Constraint
        Constraints over the variable ``y := log(|x|)``, which define this PolyDomain.

    gts : list of callable
        Inequality constraint functions (``g(x) >= 0``) which can be used to represent ``X``.

    eqs : list of callable
        Equality constraint functions (``g(x) == 0``) which can be used to represent ``X``.

    check_feas : bool
        Whether or not to check that ``X`` is nonempty. Defaults to True.

    log_AbK : tuple
        Specify a convex set in the coniclifts standard. ``log_AbK[0]`` is a SciPy sparse
        matrix. The first ``n`` columns of this matrix correspond to the variables over
        which this set is supposed to be defined. Any remaining columns are for auxiliary
        variables.

    Notes
    -----
    The constraint functions in ``gts`` and ``eqs`` should allow arguments where some components
    equal to zero. These functions can be Polynomial objects, but are not required to be.

    Only one of ``log_AbK`` and ``logspace_cons`` can be provided upon construction.
    If more than one of these value is provided, the constructor will raise an error.
    """

    __VALID_KWARGS__ = {'gts', 'eqs', 'log_AbK', 'logspace_cons', 'check_feas'}

    def __init__(self, n, **kwargs):
        for kw in kwargs:
            if kw not in PolyDomain.__VALID_KWARGS__:  # pragma: no cover
                msg = 'Provided keyword argument "' + kw + '" is not in the list'
                msg += ' of allowed keyword arguments: \n'
                msg += '\t ' + str(PolyDomain.__VALID_KWARGS__)
        self.n = n
        self.A = None
        self.b = None
        self.K = None
        self.gts = kwargs['gts'] if 'gts' in kwargs else None
        self.eqs = kwargs['eqs'] if 'eqs' in kwargs else None
        self.check_feas = kwargs['check_feas'] if 'check_feas' in kwargs else True
        self._logspace_cons = None  # optional
        self._y = None  # optional
        self._variables = None  # optional
        if 'log_AbK' in kwargs:
            self.A, self.b, self.K = kwargs['log_AbK']
            if self.check_feas:
                self._check_feasibility()
        if 'logspace_cons' in kwargs:
            if self.A is not None:
                msg = 'Keyword arguments "log_AbK" and "logspace_cons" are mutually exclusive.'
                raise RuntimeError(msg)
            else:
                self.parse_coniclifts_constraints(kwargs['logspace_cons'])
        pass

    def _check_feasibility(self):
        A, b, K = self.A, self.b, self.K
        y = cl.Variable(shape=(A.shape[1],), name='y')
        cons = [cl.PrimalProductCone(A @ y + b, K)]
        prob = cl.Problem(cl.MIN, cl.Expression([0]), cons)
        prob.solve(verbose=False, solver='ECOS')
        if not prob.value < 1e-7:
            if prob.value is np.NaN:  # pragma: no cover
                msg = 'PolyDomain constraints could not be verified as feasible.'
                msg += '\n Proceed with caution!'
                warnings.warn(msg)
            else:
                msg1 = 'PolyDomain constraints seem to be infeasible.\n'
                msg2 = 'Feasibility problem\'s status: ' + prob.status + '\n'
                msg3 = 'Feasibility problem\'s  value: ' + str(prob.value) + '\n'
                msg4 = 'The objective was "minimize 0"; we expect problem value < 1e-7. \n'
                msg = msg1 + msg2 + msg3 + msg4
                raise RuntimeError(msg)
        pass

    def check_membership(self, x_val, tol):
        """
        Evaluate ``self.gts`` and ``self.eqs`` at ``x_val``,
        to check if ``x_val`` belongs to this PolyDomain.

        Parameters
        ----------
        x_val : ndarray
            Check if ``x_val`` belongs in this domain.
        tol : float
            Infeasibility tolerance.

        Returns
        -------
        res : bool
            True iff ``x_val`` belongs to the domain represented by ``self``, up
            to infeasibility tolerance ``tol``.
        """
        if any([g(x_val) < -tol for g in self.gts]):
            return False
        if any([abs(g(x_val)) > tol for g in self.eqs]):
            return False
        return True

    def parse_coniclifts_constraints(self, logspace_cons):
        """
        Modify this PolyDomain object, so that it the log of its intersection
        with the positive orthant is the set of values satisfying constraints
        in logspace_cons.

        Parameters
        ----------
        logspace_cons : list of coniclifts.Constraint
            The provided constraints must be defined over a single coniclifts Variable.

        """
        variables = cl.compilers.find_variables_from_constraints(logspace_cons)
        if len(variables) != 1:
            raise RuntimeError('The system of constraints must be defined over a single Variable object.')
        self._logspace_cons = logspace_cons
        self._y = variables[0]
        if self._y.size != self.n:
            msg = 'The provided constraints are over a variable of dimension '
            msg += str(self._y.size) + ', but this SigDomain was declared as dimension ' + str(self.n) + '.'
            raise RuntimeError(msg)
        A, b, K, variable_map, all_variables, _ = cl.compile_constrained_system(logspace_cons)
        A = A.toarray()
        selector = variable_map[self._y.name].ravel()
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
