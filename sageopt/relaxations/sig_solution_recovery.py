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
from sageopt.symbolic.signomials import Signomial, standard_sig_monomials
from sageopt.relaxations.symbolic_correspondences import moment_reduction_array
import numpy as np
from scipy.optimize import fmin_cobyla


def is_feasible(x, greater_than_zero, equal_zero, ineq_tol=1e-8, eq_tol=1e-8, exp_format=True):
    if any([g(x, exp_format) < -ineq_tol for g in greater_than_zero]):
        return False
    if any([abs(g(x, exp_format)) > eq_tol for g in equal_zero]):
        return False
    return True


def local_refinement(f, gts, eqs, x0, rhobeg=1, rhoend=1e-7, maxfun=10000):
    res = fmin_cobyla(f, x0, gts + eqs + [-g for g in eqs],
                      rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    return res


def dual_solution_recovery(prob, ineq_tol=1e-8, eq_tol=1e-6):
    con = prob.user_cons[0]
    if not con.name == 'Lagrangian SAGE dual constraint':
        raise RuntimeError('Unexpected first constraint in dual SAGE relaxation.')
    # Recover any constraints present in "prob"
    gts, eqs = [], []
    if 'gts' in prob.associated_data:
        gts = prob.associated_data['gts']
        eqs = prob.associated_data['eqs']
    f = prob.associated_data['f']
    # Search for solutions which meet the feasibility criteria
    v = con.v.value()
    if np.any(np.isnan(v)):
        return None
    if not hasattr(con, 'alpha'):
        alpha = con.lifted_alpha[:, :con.n]
    else:
        alpha = con.alpha
    dummy_modulated_lagrangian = Signomial(alpha, np.ones(shape=(alpha.shape[0],)))
    lagrangian = _make_dummy_lagrangian(f, gts, eqs)
    alpha_reduced = lagrangian.alpha
    modulator = prob.associated_data['modulator']
    M = moment_reduction_array(lagrangian, modulator, dummy_modulated_lagrangian)
    mus0 = _least_squares_solution_recovery(alpha_reduced, con, v, M, gts, eqs, ineq_tol, eq_tol)
    mus1 = _dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol)
    mus = mus0 + mus1
    if len(mus) == 0:
        return None
    else:
        # sort the solutions according to objective quality
        mus.sort(key=lambda mu: f(mu))
        return mus


def _least_squares_solution_recovery(alpha_reduced, con, v, M, gts, eqs, ineq_tol, eq_tol):
    v_reduced = M @ v
    log_v_reduced = np.log(v_reduced)
    if isinstance(con, cl.DualCondSageCone):
        mu_ls = _constrained_least_squares(con, alpha_reduced, log_v_reduced)
    else:
        try:
            mu_ls = np.linalg.lstsq(alpha_reduced, log_v_reduced, rcond=None)[0]
        except np.linalg.linalg.LinAlgError:
            mu_ls = None
    if mu_ls is not None and is_feasible(mu_ls, gts, eqs, ineq_tol, eq_tol, exp_format=True):
        return [mu_ls]
    else:
        return []


def _constrained_least_squares(con, alpha, log_v):
    A, b, K = con.A, con.b, con.K
    A = np.asarray(A)
    x = cl.Variable(shape=(A.shape[1],))
    t = cl.Variable(shape=(1,))
    cons = [cl.vector2norm(log_v - alpha @ x[:con.n]) <= t,
            cl.PrimalProductCone(A @ x + b, K)]
    prob = cl.Problem(cl.MIN, t, cons)
    cl.clear_variable_indices()
    res = prob.solve(verbose=False)
    if res[0] == cl.SOLVED:
        mu_ls = x.value()[:con.n]
        return mu_ls
    else:
        return None


def _dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol):
    mus_exist = list(con.mu_vars.keys())
    # build a matrix whose columns are simple candidate solutions to an optimization problem.
    raw_xs = []
    for i in mus_exist:
        mu_i = con.mu_vars[i]
        xi = (mu_i.value() / v[i]).reshape((-1, 1))
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
    if isinstance(con, cl.DualCondSageCone) and min(ineq_tol, eq_tol) < 1e-5:
        # check for feasibility with respect to functional constraints, and solve an
        # optimization problem to check feasiblity with respect to conic constraints
        A, b, K = con.A, con.b, con.K
        A = np.asarray(A)
        for xi in all_xs.T:
            if is_feasible(xi, gts, eqs, ineq_tol, eq_tol, exp_format=True):
                if _satisfies_AbK_constraints(A, b, K, xi, ineq_tol):
                    mus.append(xi)
        return mus
    else:
        # only check feasibility with respect to functional constraints
        for xi in all_xs.T:
            if is_feasible(xi, gts, eqs, ineq_tol, eq_tol, exp_format=True):
                mus.append(xi)
        return mus


def _satisfies_AbK_constraints(A, b, K, mu, ineq_tol):
    # This function cannot be trusted when some component mu[i] == -np.inf.
    if np.any(np.isnan(mu)):
        return False
    x = cl.Variable(shape=(A.shape[1],))
    t = cl.Variable(shape=(1,))
    mu_flat = mu.ravel()
    where_finite = np.where(mu > -np.inf)[0]
    cons = [cl.vector2norm(mu_flat[where_finite] - x[where_finite]) <= t, cl.PrimalProductCone(A @ x + b, K)]
    prob = cl.Problem(cl.MIN, t, cons)
    cl.clear_variable_indices()
    res = prob.solve(verbose=False)
    if res[0] == cl.SOLVED and res[1] <= ineq_tol + 1e-8:
        return True
    else:
        return False


def _make_dummy_lagrangian(f, gts, eqs):
    dummy_gamma = cl.Variable(shape=())
    dummy_slacks = cl.Variable(shape=(len(gts),))
    dummy_multipliers = cl.Variable(shape=(len(eqs)))
    ineq_term = sum([gts[i] * dummy_slacks[i] for i in range(len(gts))])
    eq_term = sum([eqs[i] * dummy_multipliers[i] for i in range(len(eqs))])
    dummy_L = f - dummy_gamma - ineq_term - eq_term
    cl.clear_variable_indices()
    return dummy_L
