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
from sageopt.symbolic.polynomials import Polynomial, standard_poly_monomials
from sageopt.relaxations.symbolic_correspondences import moment_reduction_array
from sageopt.relaxations.sig_solution_recovery import is_feasible, _make_dummy_lagrangian
import numpy as np
from scipy.optimize import fmin_cobyla
import itertools


#TODO: update this function to allow SigDomain arguments.
#   Bring back "log_domain_converter" to turn the SigDomain into a PolyDomain
def local_refine_polys_from_sigs(f, gts, eqs, x0, **kwargs):
    """
    This is a helper function which ...
        (1) accepts signomial problem data (representative of a desired polynomial optimization problem),
        (2) transforms the signomial data into equivalent polynomial data, and
        (3) performs local refinement on the polynomial data, via the COBYLA solver.

    Parameters
    ----------

    f: Signomial
        Defines the objective function to be minimized. From "f" we will construct
        a polynomial "p" where ``p(y) = f(np.log(y))`` for all positive vectors y.

    gts : list of Signomial
        Each defining an inequality constraint ``g(x) >= 0``. From this list, we will construct a
        list of polynomials gts_poly, so that every ``g0 in gts`` has a polynomial representative
        ``g1 in gts_poly``, satisfying ``g1(y) = g0(np.log(y))`` for all positive vectors y.

    eqs : list of Signomial
        Each defining an equality constraint ``g(x) == 0``. From this list, we will construct a
        list of polynomials ``eqs_poly``, so that every ``g0 in gts`` has a polynomial representative
        ``g1 in eqs_poly``, satisfying ``g1(y) = g0(np.log(y))`` for all positive vectors y.

    x0 : ndarray
        An initial condition for the *signomial* optimization problem
        ``min{ f(x) |  g(x) >= 0 for g in gts, g(x) == 0 for g in eqs }``.

    Other Parameters
    ----------------

    rhobeg : float
        Controls the size of COBYLA's initial search space around ``y0 = exp(x0)``.

    rhoend : float
        Termination criteria, controlling the size of COBYLA's smallest search space.

    maxfun : int
        Termination criteria, bounding the number of COBYLA's iterations.

    Returns
    -------
    y : ndarray

        The output of COBYLA for the polynomial optimization problem
        ``min{ p(y) | g(y) >= 0 for g in gts_poly, g(y) == 0 for g in eqs_poly, y >= 0 }``
        with initial condition ``y0 = exp(x0)``.

    """
    rhobeg = kwargs['rhobeg'] if 'rhobeg' in kwargs else 1.0
    rhoend = kwargs['rhoend'] if 'rhoend' in kwargs else 1e-7
    maxfun = int(kwargs['maxfun']) if 'maxfun' in kwargs else 10000
    y0 = np.exp(x0)
    gts = [g.as_polynomial() for g in gts]
    x = standard_poly_monomials(y0.size)
    gts += [x[i] for i in range(y0.size)]  # Decision variables must be nonnegative.
    eqs = [g.as_polynomial() for g in eqs]
    f = f.as_polynomial()
    y = fmin_cobyla(f, y0, gts + eqs + [-g for g in eqs],
                    rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    return y


#TODO: update this function to work with poly_relaxation
def poly_solrec(prob, ineq_tol=1e-8, eq_tol=1e-6, skip_ls=False, **kwargs):
    """
    Recover a list of candidate solutions from a dual SAGE relaxation. Solutions are
    guaranteed to be feasible up to specified tolerances, but not necessarily optimal.

    Parameters
    ----------
    prob : coniclifts.Problem
        A dual-form SAGE relaxation, from ``poly_constrained_relaxation``.

    ineq_tol : float
        The amount by which recovered solutions can violate inequality constraints.

    eq_tol : float
        The amount by which recovered solutions can violate equality constraints.

    skip_ls : bool
        Whether or not to skip least-squares solution recovery.

    Returns
    -------
    sols : list of ndarrays
        A list of feasible solutions, sorted in increasing order of objective function value.
        It is possible that this list is empty, in which case no feasible solutions were recovered.

    Notes
    -----
    This function accepts the following keyword arguments:

    zero_tol : float
        Used in magnitude recovery. If a component of the Lagrangian's moment vector is smaller
        than this (in absolute value), pretend it's zero in the least-squares step. Defaults to 1e-20.

    heuristic_signs : bool
        Used in sign recovery. If True, then attempts to infer variable signs from the Lagrangian's
        moment vector even when a completely consistent set of signs does not exist. Defaults to True.

    all_signs : bool
        Used in sign recovery. If True, then consider returning solutions which differ only by sign.
        Defaults to True.

    This function is implemented only for poly_constrained_relaxation (not poly_relaxation).
    """
    zero_tol = kwargs['zero_tol'] if 'zero_tol' in kwargs else 1e-20
    heuristic = kwargs['heuristic_signs'] if 'heuristic_signs' in kwargs else True
    all_signs = kwargs['all_signs'] if 'all_signs' in kwargs else True
    metadata = prob.metadata
    f = metadata['f']
    lag_gts = metadata['gts']
    lag_eqs = metadata['eqs']
    lagrangian = _make_dummy_lagrangian(f, lag_gts, lag_eqs)
    con = prob.constraints[0]
    alpha = con.alpha
    dummy_modulated_lagrangian = Polynomial(alpha, np.ones(shape=(alpha.shape[0],)))  # coefficients dont matter
    modulator = metadata['modulator']
    v = metadata['v_poly'].value  # possible that v_sig and v are the same
    if np.any(np.isnan(v)):
        return []
    M = moment_reduction_array(lagrangian, modulator, dummy_modulated_lagrangian)
    v_reduced = M @ v
    alpha_reduced = lagrangian.alpha
    mags = variable_magnitudes(con, alpha_reduced, v_reduced, zero_tol, skip_ls)
    signs = variable_sign_patterns(alpha_reduced, v_reduced, heuristic, all_signs)
    # Now we need to build the candidate solutions, and check them for feasibility.
    if con.X is not None:
        gts = lag_gts + [g for g in con.X.gts]
        eqs = lag_eqs + [g for g in con.X.eqs]
    else:
        gts = lag_gts
        eqs = lag_eqs
    solutions = []
    for mag in mags:
        for sign in signs:
            x = mag * sign  # elementwise
            if is_feasible(x, gts, eqs, ineq_tol, eq_tol):
                solutions.append(x)
    solutions.sort(key=lambda xi: f(xi))
    return solutions


def variable_magnitudes(con, alpha_reduced, v_reduced, zero_tol, skip_ls):
    # This is essentially "Algorithm 3" in the conditional SAGE paper.
    v_sig = con.v.value
    M_sig = np.eye(v_sig.size)
    mags0 = _dual_age_cone_magnitude_recovery(con, v_sig, M_sig)
    if skip_ls:
        return mags0
    else:
        abs_mom_mag = _least_squares_magnitude_recovery(con, alpha_reduced, v_reduced, zero_tol)
        if abs_mom_mag is None:
            mags1 = []
        else:
            mags1 = [abs_mom_mag]
        mags = mags0 + mags1
        return mags


def _dual_age_cone_magnitude_recovery(con, v_sig, M_sig):
    mus_exist = list(con.mu_vars.keys())
    # build a matrix whose columns are simple candidate solutions to an optimization problem.
    raw_ys = []
    for i in mus_exist:
        mu_i = con.mu_vars[i]
        yi = (mu_i.value / v_sig[i]).reshape((-1, 1))
        raw_ys.append(yi)
    raw_ys = np.hstack(raw_ys)
    # build a matrix "weights", whose rows specify convex combination coefficients.
    M_interest = M_sig[:, mus_exist]
    v_interest = v_sig[mus_exist].ravel()
    v_reduced = M_interest @ v_interest
    keep_rows = v_reduced != 0
    M_interest = M_interest[keep_rows, :]
    v_reduced = v_reduced[keep_rows]
    weights = M_interest / v_reduced[:, None]
    weights = (weights.T * v_interest[:, None]).T
    # the rows of "weights" now define convex combination vectors.
    reduced_ys = raw_ys @ weights.T
    # concatenate the new solutions (reduced_ys) with original solutions (raw_ys)
    all_ys = np.hstack((raw_ys, reduced_ys))
    all_ys = np.unique(all_ys, axis=1).astype(np.longdouble)
    all_xs = np.exp(all_ys)
    mus = [xi for xi in all_xs.T]
    return mus


def _least_squares_magnitude_recovery(con, alpha_reduced, v_reduced, zero_tol):
    v_abs = np.abs(v_reduced).ravel()
    if con.X is not None:
        n = con.X.A.shape[1]
    else:
        n = con.alpha.shape[1]
    if n > con.alpha.shape[1]:
        padding = np.zeros(shape=(alpha_reduced.shape[0], n - con.alpha.shape[1].n))
        alpha_reduced = np.hstack((alpha_reduced, padding))
    y = cl.Variable(shape=(n,), name='abs moment mag recovery')
    are_nonzero = v_abs > np.sqrt(zero_tol)
    t = cl.Variable(shape=(1,), name='t')
    residual = alpha_reduced[are_nonzero, :] @ y - np.log(v_abs[are_nonzero])
    constraints = [cl.vector2norm(residual) <= t]
    if np.any(~are_nonzero):
        tempcon = alpha_reduced[~are_nonzero, :] @ y <= np.log(zero_tol)
        constraints.append(tempcon)
    if con.X is not None:
        A, b, K = con.X.A, con.X.b, con.X.K
        tempcon = cl.PrimalProductCone(A @ y + b, K)
        constraints.append(tempcon)
    prob = cl.Problem(cl.MIN, t, constraints)
    prob.solve(verbose=False)
    cl.clear_variable_indices()
    if prob.status in {cl.SOLVED, cl.INACCURATE} and prob.value < np.inf:
        mag = np.exp(y.value.astype(np.longdouble))
        return mag
    else:
        return None


def variable_sign_patterns(alpha, moments, hueristic=False, all_signs=True):
    # This is essentially "Algorithm 4" in the conditional SAGE paper.
    #
    # ignore signs of variables which only participate in even monomials.
    #
    # variables that participate in monomials "j" with moments[j] == 0
    # are given nonnegative signs.
    m, n = alpha.shape
    x0, alpha1, U, W = linear_system_negatives(alpha, moments)
    if x0 is None:
        if not hueristic:
            return []
        else:
            x0 = greedy_weighted_cut_negatives(alpha, moments)
            y0 = np.ones(n,)
            y0[x0 == 1] = -1
            return [y0]
    elif alpha1 is None:
        y0 = np.ones(n,)
        return [y0]
    else:
        if all_signs:
            arref, p = mod2rref(alpha1)
            N0 = mod2nullspace(arref, p)
        else:
            N0 = [np.zeros(alpha1.shape[1],)]
        signs = []
        for vec0 in N0:
            vec = np.zeros(n,)
            vec[W] = vec0
            vec = np.mod(vec + x0, 2).astype(int)
            y = np.ones(n,)
            y[vec == 1] = -1
            signs.append(y)
        return signs


def linear_system_negatives(alpha, moments):
    """
    :param alpha: an m-by-n array of integers.
    :param moments: a vector of length m.
    :return: a length-n vector with entries from {0, 1}. 1 means "negative",
    "0" means "nonnegative".
    """
    m, n = alpha.shape
    alpha = np.mod(alpha, 2).astype(int)
    U = [i for i in range(m) if abs(moments[i]) > 0 and np.any(alpha[i, :] > 0)]
    # If moment[i] == 0, then there is no sign we need to be consistent with.
    # If alpha[i,:] == 0 (after taking alpha mod 2), then we must have moments[i] >= 0,
    # and the resulting row of the linear system is trivially satisfied. Therefore
    # we restrict our attention to rows "i" with some alpha[i, j] = 1 (mod 2) for some
    # j, without loss of generality.
    if len(U) == 0:
        return np.zeros(n,), None, None, None
    W = [j for j in range(n) if np.any(alpha[U, j] > 0)]
    # If, out of the remaining rows in the linear system, the variable at coordinate "j"
    # only participates in even monomials, then we remove these columns from the matrix
    # "alpha". Removing these columns has a very modest speed improvement here (when
    # finding a single solution to a linear system), but can have a very large effect
    # when enumerating all vectors in the mod-2 nullspace of the smaller matrix.
    if len(W) == 0:
        return np.zeros(n,), None, None, None
    alpha = alpha[U, :]
    alpha = alpha[:, W]
    b = ((moments < 0)[U]).astype(int)
    x_W = mod2linsolve(alpha, b)
    if x_W is None:
        return None, alpha, U, W
    else:
        x = np.zeros(n,)
        x[W] = x_W
        return x, alpha, U, W


def greedy_weighted_cut_negatives(alpha, v):
    m, n = alpha.shape
    y = np.ones(n,)
    v = v.ravel()
    coordinates_remaining = set(range(n))
    merit_of_switch = np.zeros(n,)
    for iter_num in range(n):
        for i in coordinates_remaining:
            before = np.dot(v, np.prod(np.power(y, alpha), axis=1))
            y[i] = -1
            after = np.dot(v, np.prod(np.power(y, alpha), axis=1))
            y[i] = 1
            merit_of_switch[i] = after - before
        best_candidate = np.argmax(merit_of_switch)
        best_candidate = best_candidate[(0,) * best_candidate.ndim]
        if merit_of_switch[best_candidate] > 0:
            # if "merit" is positive, then the vector y^alpha
            # points "more" along the direction of "v" than it
            # did before. Ideally we want y^alpha to point exactly
            # along sign(v), but the intended use-case of this function
            # is when no such value of y in {+1, 1}^n can have y^alpha
            # match sign(v). Therefore it makes more sense to use a
            # weighted measure of similarity -- taking the inner-product
            # of v and y^alpha is a reasonable measure of similarity.
            y[best_candidate] = -1
            merit_of_switch[best_candidate] = -1
            coordinates_remaining.remove(best_candidate)
        else:
            break
    x = y.copy()
    x[y == -1] = 1
    x[y == 1] = 0
    return x


def mod2rref(A, forward_only=False):
    """
    :param A: an integer array with A.shape == (m, n).
    :param forward_only: boolean. False means "return reduced row-echelon form", True means
    "return row echelon form".
    :return: the (reduced) row-echelon form of A (in mod 2 arithmetic) and corresponding pivot columns
    """
    A = np.mod(A, 2).astype(int)
    h, k = 0, 0
    m, n = A.shape
    pivot_columns = []
    # Forward elimination
    while h < m and k < n:
        i_max = h + np.argmax(A[h:, k])
        if A[i_max, k] == 0:
            k += 1
        else:
            # swap rows "h" and "i_max"
            pivot_columns.append(k)
            row_h = A[h, :].copy()
            row_imax = A[i_max, :].copy()
            A[h, :] = row_imax
            A[i_max, :] = row_h
            for i in range(h+1, m):
                if A[i, k] > 0:
                    A[i, :] = np.mod(A[i, :] - A[h, :], 2)
            h += 1
            k += 1
    # Back substitution
    if not forward_only:
        for pr, pc in enumerate(pivot_columns):
            for row in range(pr-1, -1, -1):
                if A[row, pc] > 0:
                    # ^ That isn't necessarily valid when a matrix is wide, and contains
                    # a pivot column far down the list of columns.
                    A[row, pc:] = np.mod(A[row, pc:] - A[pr, pc:], 2)
    return A, pivot_columns


def mod2nullspace_basis(arref, p):
    m, n = arref.shape
    r = len(p)
    F = {j for j in range(n) if j not in p}  # set of free variables
    N = np.zeros(shape=(n, len(F)), dtype=int)
    for ell, f in enumerate(F):
        N[f, ell] = 1
        for i in range(r):
            N[p[i], ell] = arref[i, f]
    return N


def mod2nullspace(arref, p):
    N = mod2nullspace_basis(arref, p)
    N = [vec for vec in N.T]
    nullspace = {(0,) * arref.shape[1]}
    powerset_iterator = itertools.chain.from_iterable(itertools.combinations(N, r) for r in range(1, len(N)+1))
    for vecs in powerset_iterator:
        vec = np.mod(np.sum(vecs, axis=0), 2).astype(int)
        vec = tuple(vec.tolist())
        nullspace.add(vec)
    N = [np.array(vec) for vec in nullspace]
    return N


def mod2linsolve(A, b):
    """
    :param A: m-by-n
    :param b: m
    :return: solution to A @ x == b (mod 2), or None.
    """
    m, n = A.shape
    A0 = np.hstack([A, b.reshape((-1, 1))])
    A1, pivcols = mod2rref(A0, forward_only=True)
    augmented_rank = len(pivcols)
    if augmented_rank > 0 and pivcols[-1] == n:
        return None
    else:
        b1 = A1[:n, A.shape[1]]
        A1 = A1[:n, :A.shape[1]]
        x = np.zeros(shape=(n,), dtype=int)
        row = len(pivcols) - 1
        for pc in reversed(pivcols):
            x[pc] = (b1[row] - np.dot(A1[row, (pc+1):], x[(pc+1):])) % 2
            row -= 1
        return x
