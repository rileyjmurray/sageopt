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
from sageopt.relaxations.sig_solution_recovery import is_feasible, local_refine
import numpy as np
from scipy.optimize import fmin_cobyla
import itertools


def local_refine_polys_from_sigs(f, gts, eqs, x0, rhobeg=1.0, rhoend=1e-7, maxfun=10000):
    """
    When faced with a polynomial optimization problem over nonnegative variables, one should
    formulate the problem in terms of signomials. This reformulation is without loss of generality
    from the perspective of solving a SAGE relaxation, but the local-refinement stage of solution
    recovery is somewhat different.

    This is a helper function which ...
        (1) accepts signomial problem data (representative of a desired polynomial optimization problem),
        (2) transforms the signomial data into equivalent polynomial data, and
        (3) performs local refinement on the polynomial data, via the COBYLA solver.

    This function may be important if a polynomial optimization problem has some variable equal to zero
    in an optimal solution.

    :param f: a Signomial, defining the objective function to be minimized. From "f" we will construct
    a polynomial "p" where p(y) = f(log(y)) for all elementwise positive vectors y.

    :param gts: a list of Signomials, each defining an inequality constraint g(x) >= 0.
    From this list, we will construct a list of polynomials gts_poly, so that every g0 in gts has a
    polynomial representative g1 in gts_poly, satisfying g1(y) = g0(log(y)) for all elementwise
    positive vectors y.

    :param eqs: a list of Signomials, each defining an equality constraint g(x) == 0.
    From this list, we will construct a list of polynomials gts_poly, so that every g0 in eqs has a
    polynomial representative g1 in eqs_poly, satisfying g1(y) = g0(log(y)) for all elementwise
    positive vectors y.

    :param x0: an initial condition for the signomial optimization problem
        min{ f(x) |  g(x) >= 0 for g in gts, g(x) == 0 for g in eqs }.

    :param rhobeg: COBYLA's initial radius of the search space around y0 = exp(x0).

    :param rhoend: COBYLA terminates once the radius of the search space reaches this value.

    :param maxfun: The maximum number of iterations used by the COBYLA solver.

    :return: The output of COBYLA for the polynomial optimization problem
        min{ p(y) | g(y) >= 0 for g in gts_poly, g(y) == 0 for g in eqs_poly, y >= 0 }
    with initial condition y0 = exp(x0).
    """
    x0 = np.exp(x0)
    gts = [g.as_polynomial() for g in gts]
    x = standard_poly_monomials(x0.size)
    gts += [x[i] for i in range(x0.size)]  # Decision variables must be nonnegative.
    eqs = [g.as_polynomial() for g in eqs]
    f = f.as_polynomial()
    res = fmin_cobyla(f, x0, gts + eqs + [-g for g in eqs],
                      rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    return res


def poly_solrec(prob, ineq_tol=1e-8, eq_tol=1e-6, zero_tol=1e-20, hueristic=False, all_signs=True):
    # implemented only for poly_constrained_dual (not yet implemented for sage_poly_dual).
    f = prob.associated_data['f']
    lag_gts = prob.associated_data['gts']
    lag_eqs = prob.associated_data['eqs']
    lagrangian = _make_dummy_lagrangian(f, lag_gts, lag_eqs)
    con = prob.user_cons[0]
    if isinstance(con, cl.DualCondSageCone):
        alpha = con.lifted_alpha[:, :con.n]
    else:
        alpha = con.alpha
    dummy_modulated_lagrangian = Polynomial(alpha, np.ones(shape=(alpha.shape[0],)))  # coefficients dont matter
    modulator = prob.associated_data['modulator']
    v = prob.associated_data['v_poly'].value  # possible that v_sig and v are the same
    if np.any(np.isnan(v)):
        return []
    M = moment_reduction_array(lagrangian, modulator, dummy_modulated_lagrangian)
    v_reduced = M @ v
    alpha_reduced = lagrangian.alpha
    mags = variable_magnitudes(con, alpha_reduced, v_reduced, zero_tol)
    signs = variable_sign_patterns(alpha_reduced, v_reduced, hueristic, all_signs)
    # Now we need to build the candidate solutions, and check them for feasibility.
    gts = lag_gts + [g for g in prob.associated_data['X']['gts']]
    eqs = lag_eqs + [g for g in prob.associated_data['X']['eqs']]
    solutions = []
    for mag in mags:
        for sign in signs:
            x = mag * sign  # elementwise
            if is_feasible(x, gts, eqs, ineq_tol, eq_tol):
                solutions.append(x)
    solutions.sort(key=lambda xi: f(xi))
    return solutions


def variable_magnitudes(con, alpha_reduced, v_reduced, zero_tol):
    # This is essentially "Algorithm 3" in the conditional SAGE paper.
    v_sig = con.v.value
    M_sig = np.eye(v_sig.size)
    mags0 = _dual_age_cone_magnitude_recovery(con, v_sig, M_sig)
    abs_mom_mag = _abs_moment_feasibility_magnitude_recovery(con, alpha_reduced, v_reduced, zero_tol)
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
    all_ys = np.unique(all_ys, axis=1).astype(np.float128)
    all_xs = np.exp(all_ys)
    mus = [xi for xi in all_xs.T]
    return mus


def _abs_moment_feasibility_magnitude_recovery(con, alpha_reduced, v_reduced, zero_tol):
    v_abs = np.abs(v_reduced).ravel()
    if isinstance(con, cl.DualCondSageCone):
        n = con.lifted_n
    else:
        n = con.n
    if n > con.n:
        padding = np.zeros(shape=(alpha_reduced.shape[0], n - con.n))
        alpha_reduced = np.hstack((alpha_reduced, padding))
    y = cl.Variable(shape=(n,), name='abs moment mag recovery')
    are_nonzero = v_abs > np.sqrt(zero_tol)
    t = cl.Variable(shape=(1,), name='t')
    if np.any(~are_nonzero):
        constraints = [alpha_reduced[are_nonzero, :] @ y <= np.log(v_abs[are_nonzero] + zero_tol),
                       np.log(v_abs[are_nonzero] - zero_tol) <= alpha_reduced[are_nonzero, :] @ y,
                       np.log(zero_tol) <= t,
                       alpha_reduced[~are_nonzero, :] @ y <= t]
    else:
        constraints = [cl.vector2norm(alpha_reduced @ y - np.log(v_abs)) <= t]
    if isinstance(con, cl.DualCondSageCone):
        A, b, K = np.asarray(con.A), con.b, con.K
        constraints.append(cl.PrimalProductCone(A @ y + b, K))
    prob = cl.Problem(cl.MIN, t, constraints)
    prob.solve(verbose=False)
    cl.clear_variable_indices()
    if prob.status == cl.SOLVED and prob.value < np.inf:
        mag = np.exp(y.value.astype(np.float128))
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


def _lin_indep_subset_negatives(alpha, b):
    _, lin_indep_row_idxs = mod2rref(alpha.T)
    alpha = alpha[lin_indep_row_idxs, :]
    b = b[lin_indep_row_idxs]
    x = mod2linsolve(alpha, b)
    return x


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


def _make_dummy_lagrangian(f, gts, eqs):
    dummy_gamma = cl.Variable(shape=())
    dummy_slacks = cl.Variable(shape=(len(gts),))
    dummy_multipliers = cl.Variable(shape=(len(eqs)))
    ineq_term = sum([gts[i] * dummy_slacks[i] for i in range(len(gts))])
    eq_term = sum([eqs[i] * dummy_multipliers[i] for i in range(len(eqs))])
    dummy_L = f - dummy_gamma - ineq_term - eq_term
    cl.clear_variable_indices()
    return dummy_L
