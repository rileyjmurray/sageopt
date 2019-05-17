import sageopt.coniclifts as cl
from sageopt.symbolic.signomials import Signomial, is_feasible, standard_sig_monomials
from sageopt.relaxations.symbolic_correspondences import moment_reduction_array
import numpy as np
import warnings
from scipy.optimize import fmin_cobyla


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
    lagrangian = prob.associated_data['lagrangian']
    modulator = prob.associated_data['modulator']
    M = moment_reduction_array(lagrangian, modulator, dummy_modulated_lagrangian)
    mus0 = _least_squares_solution_recovery(lagrangian, con, v, M, gts, eqs, ineq_tol, eq_tol)
    mus1 = _dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol)
    mus = mus0 + mus1
    if len(mus) == 0:
        return None
    else:
        # sort the solutions according to objective quality
        mus.sort(key=lambda mu: f(mu))
        mus = np.hstack(mus)
        return mus


def _least_squares_solution_recovery(lagrangian, con, v, M, gts, eqs, ineq_tol, eq_tol):
    v_reduced = M @ v
    log_v_reduced = np.log(v_reduced)
    alpha = lagrangian.alpha
    if isinstance(con, cl.DualCondSageCone):
        mu_ls = _constrained_least_squares(con, alpha, log_v_reduced)
    else:
        try:
            mu_ls = np.linalg.lstsq(alpha, log_v_reduced, rcond=None)[0].reshape((-1, 1))
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
        mu_ls = x.value()[:con.n].reshape((-1, 1))
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
    if isinstance(con, cl.DualCondSageCone) and ineq_tol < 1e-5:
        # check for feasibility with respect to functional constraints, and solve an
        # optimization problem to check feasiblity with respect to conic constraints
        A, b, K = con.A, con.b, con.K
        A = np.asarray(A)
        for xi in all_xs.T:
            if is_feasible(xi, gts, eqs, ineq_tol, eq_tol, exp_format=True):
                if _satisfies_AbK_constraints(A, b, K, xi, ineq_tol):
                    mus.append(xi.reshape((-1, 1)))
        return mus
    else:
        # only check feasibility with respect to functional constraints
        for xi in all_xs.T:
            if is_feasible(xi, gts, eqs, ineq_tol, eq_tol, exp_format=True):
                mus.append(xi.reshape((-1, 1)))
        return mus


def _satisfies_AbK_constraints(A, b, K, mu, ineq_tol):
    if np.any(np.isnan(mu)):
        return False
    x = cl.Variable(shape=(A.shape[1],))
    t = cl.Variable(shape=(1,))
    mu_flat = mu.ravel()
    cons = [cl.vector2norm(mu_flat - x[:mu.size]) <= t, cl.PrimalProductCone(A @ x + b, K)]
    prob = cl.Problem(cl.MIN, t, cons)
    cl.clear_variable_indices()
    res = prob.solve(verbose=False)
    if res[0] == cl.SOLVED and res[1] <= ineq_tol + 1e-8:
        return True
    else:
        return False
