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
from sageopt.symbolic.signomials import Signomial
from sageopt.relaxations.symbolic_correspondences import moment_reduction_array
import numpy as np
from scipy.optimize import fmin_cobyla


def is_feasible(x, greater_than_zero, equal_zero, ineq_tol=1e-8, eq_tol=1e-8):
    if any([g(x) < -ineq_tol for g in greater_than_zero]):
        return False
    if any([abs(g(x)) > eq_tol for g in equal_zero]):
        return False
    return True


def local_refine(f, gts, eqs, x0, rhobeg=1, rhoend=1e-7, maxfun=1e4):
    """
    Use SciPy's COBYLA solver in an attempt to find a minimizer of ``f`` subject to
    inequality constraints in ``gts`` and equality constraints in ``eqs``.

    Parameters
    ----------
    f : a callable function
        The minimization objective.
    gts : a list of callable functions
        Each ``g in gts`` specifies an inequality constraint ``g(x) >= 0``.
    eqs : a list of callable functions
        Each ``g in eqs`` specifies an equality constraint ``g(x) == 0``.
    x0 : ndarray
        An initial point for COBYLA.
    rhobeg : float
        Controls the size of COBYLA's initial search space.
    rhoend : float
        Termination criteria, controlling the size of COBYLA's smallest search space.
    maxfun : int
        Termination criteria, bounding the number of COBYLA's iterations.

    Returns
    -------
    x : ndarray
        The solution returned by COBYLA.

    """
    maxfun = int(maxfun)
    x = fmin_cobyla(f, x0, gts + eqs + [-g for g in eqs],
                      rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    return x


def sig_solrec(prob, ineq_tol=1e-8, eq_tol=1e-6, skip_ls=False):
    """
    Recover a list of candidate solutions from a dual SAGE relaxation. Solutions are
    guaranteed to be feasible up to specified tolerances, but not necessarily optimal.

    Parameters
    ----------
    prob : coniclifts.Problem
        A dual-form SAGE relaxation.
    ineq_tol : float
        The amount by which recovered solutions can violate inequality constraints.
    eq_tol : float
        The amount by which recovered solutions can violate equality constraints.
    skip_ls : bool
        Whether or not to skip constrained least-squares solution recovery.

    Returns
    -------
    sols : list of ndarrays
        A list of feasible solutions, sorted in increasing order of objective function value.
        It is possible that this list is empty, in which case no feasible solutions were recovered.

    """
    con = prob.constraints[0]
    if not con.name == 'Lagrangian SAGE dual constraint':  # pragma: no cover
        raise RuntimeError('Unexpected first constraint in dual SAGE relaxation.')
    metadata = prob.metadata
    f = metadata['f']
    # Recover any constraints present in "prob"
    lag_gts, lag_eqs = [], []
    if 'gts' in metadata:
        # only happens in "constrained_sage_dual".
        lag_gts = metadata['gts']
        lag_eqs = metadata['eqs']
    lagrangian = _make_dummy_lagrangian(f, lag_gts, lag_eqs)
    if con.X is None:
        X_gts, X_eqs = [], []
    else:
        X_gts, X_eqs = con.X.gts, con.X.eqs
    gts = lag_gts + X_gts
    eqs = lag_eqs + X_eqs
    # Search for solutions which meet the feasibility criteria
    v = con.v.value
    v[v < 0] = 0
    if np.any(np.isnan(v)):
        return None
    alpha = con.alpha
    dummy_modulated_lagrangian = Signomial(alpha, np.ones(shape=(alpha.shape[0],)))
    alpha_reduced = lagrangian.alpha
    modulator = metadata['modulator']
    M = moment_reduction_array(lagrangian, modulator, dummy_modulated_lagrangian)
    if skip_ls:
        sols0 = []
    else:
        sols0 = _least_squares_solution_recovery(alpha_reduced, con, v, M, gts, eqs, ineq_tol, eq_tol)
    sols1 = _dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol)
    sols = sols0 + sols1
    sols.sort(key=lambda mu: f(mu))
    return sols


def _least_squares_solution_recovery(alpha_reduced, con, v, M, gts, eqs, ineq_tol, eq_tol):
    v_reduced = M @ v
    log_v_reduced = np.log(v_reduced + 1e-8)
    if con.X is not None:
        mu_ls = _constrained_least_squares(con, alpha_reduced, log_v_reduced)
    else:
        try:
            mu_ls = np.linalg.lstsq(alpha_reduced, log_v_reduced, rcond=None)[0]
        except np.linalg.linalg.LinAlgError:
            mu_ls = None
    if mu_ls is not None and is_feasible(mu_ls, gts, eqs, ineq_tol, eq_tol):
        return [mu_ls]
    else:
        return []


def _constrained_least_squares(con, alpha, log_v):
    A, b, K = con.X.A, con.X.b, con.X.K
    lifted_n = A.shape[1]
    n = con.alpha.shape[1]
    x = cl.Variable(shape=(lifted_n,))
    t = cl.Variable(shape=(1,))
    cons = [cl.vector2norm(log_v - alpha @ x[:n]) <= t,
            cl.PrimalProductCone(A @ x + b, K)]
    prob = cl.Problem(cl.MIN, t, cons)
    cl.clear_variable_indices()
    res = prob.solve(verbose=False)
    if res[0] in {cl.SOLVED, cl.INACCURATE}:
        mu_ls = x.value[:n]
        return mu_ls
    else:
        return None


def _dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol):
    # TODO: refactor this method to consider solution recovery of the form
    #   x_candidate = sum([ con.mu_vars[i] for i in I]) / sum(con.v[I])
    #   for arbitrary index sets I.
    #   Right now we're doing this in a much more complicated way, by first
    #   forming x_candidates for I = {i} (singletons), then then taking
    #   convex combinations. This current approach may be subject to bad
    #   rounding errors if some con.v[i] are near zero.
    mus_exist = list(con.mu_vars.keys())
    mus_exist = [i for i in mus_exist if v[i] > 0]
    if len(mus_exist) == 0:
        return []
    # build a matrix whose columns are simple candidate solutions to an optimization problem.
    raw_xs = []
    for i in mus_exist:
        mu_i = con.mu_vars[i]
        xi = (mu_i.value / v[i]).reshape((-1, 1))
        raw_xs.append(xi)
    raw_xs = np.hstack(raw_xs)
    # build a matrix "weights", whose rows specify convex combination coefficients.
    M_interest = M[:, mus_exist]
    v_interest = v[mus_exist].ravel()
    v_reduced = M_interest @ v_interest
    keep_rows = v_reduced != 0
    M_interest = M_interest[keep_rows, :]
    v_reduced = v_reduced[keep_rows]
    weights = M_interest / v_reduced[:, None]
    weights = (weights.T * v_interest[:, None]).T
    # the rows of "weights" now define convex combination vectors.
    reduced_xs = raw_xs @ weights.T
    # concatenate the new solutions (reduced_xs) with original solutions (raw_xs)
    all_xs = np.hstack((raw_xs, reduced_xs))
    all_xs = np.unique(all_xs, axis=1)
    # check each candidate solution for feasibility (up to desired tolerances)
    # and return the result
    mus = []
    for xi in all_xs.T:
        if is_feasible(xi, gts, eqs, ineq_tol, eq_tol):
            mus.append(xi)
    return mus


def _make_dummy_lagrangian(f, gts, eqs):
    dummy_gamma = cl.Variable(shape=())
    if len(gts) > 0:
        dummy_slacks = cl.Variable(shape=(len(gts),))
        ineq_term = sum([gts[i] * dummy_slacks[i] for i in range(len(gts))])
    else:
        ineq_term = 0
    if len(eqs) > 0:
        dummy_multipliers = cl.Variable(shape=(len(eqs),))
        eq_term = sum([eqs[i] * dummy_multipliers[i] for i in range(len(eqs))])
    else:
        eq_term = 0
    dummy_L = f - dummy_gamma - ineq_term - eq_term
    cl.clear_variable_indices()
    return dummy_L

