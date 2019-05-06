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
from itertools import combinations_with_replacement
from sageopt.symbolic.signomials import Signomial
from sageopt import coniclifts as cl
from sageopt.relaxations import symbolic_correspondences as sym_corr
import warnings


def up_to_q_fold_cons(cons, q):
    cons = [g for g in cons]  # shallow copy
    if len(cons) == 0 or q == 1:
        return cons
    else:
        fold_cons = set()
        for qq in range(q + 1):
            if qq == 0:
                continue
            for comb in combinations_with_replacement(cons, qq):
                temp = np.prod(comb)
                # noinspection PyUnresolvedReferences
                if np.count_nonzero(temp.c) > 1:
                    fold_cons.add(temp)
        fold_cons = list(fold_cons)
    return fold_cons


def valid_posynomial_inequalities(gs):
    conv_gs = []
    for g in gs:
        # g defines a constraint g(x) >= 0
        num_pos = np.count_nonzero(g.c > 0)
        if num_pos >= 2:
            # cannot convexify
            continue
        elif num_pos == 0:
            raise RuntimeError('Attempting to convexify an infeasible signomial inequality constraint.')
        else:
            pos_loc = np.where(g.c > 0)[0][0]
            inverse_term = Signomial({tuple(-g.alpha[pos_loc, :]): 1})
            conv_gs.append(g * inverse_term)
    return conv_gs


def valid_monomial_equations(eqs):
    conv_eqs = []
    for g in eqs:
        # g defines a constraint g(x) == 0.
        if np.count_nonzero(g.c) > 2:
            # cannot convexify
            continue
        pos_loc = np.where(g.c > 0)[0]
        if pos_loc.size == 1:
            pos_loc = pos_loc[0]
            inverse_term = Signomial({tuple(-g.alpha[pos_loc, :]): 1})
            conv_eqs.append(g * inverse_term)
    return conv_eqs


def valid_gp_representable_poly_inequalities(gs):
    gp_rep_polys = []
    for g in gs:
        is_even = len(g.even_locations()) == g.m
        num_pos = np.count_nonzero(g.c > 0)
        if num_pos == 1 and is_even:
            gp_rep_polys.append(g)
        elif num_pos == 0 and is_even:
            if g(np.zeros(g.n)) == 0:
                warnings.warn('The polynomial "g" is only nonnegative at the origin.')
            else:
                raise RuntimeError('Infeasible polynomial inequality.')
    return gp_rep_polys


def valid_gp_representable_poly_eqs(eqs):
    gp_rep_polys = []
    for g in eqs:
        is_even = len(g.even_locations()) == g.m
        if is_even and np.count_nonzero(g.c != 0) == 2 and np.count_nonzero(g.c > 0) == 1:
            gp_rep_polys.append(g)
    return gp_rep_polys


def unconstrained_cut(v, alpha, ell=1, AbK=None, solver='MOSEK'):
    # v is supposed to belong to convco{ exp(alpha @ x) : x in Rn }, but instead
    # it only belongs to some outer-approximation. We need to find a separating hyperplane
    # (in terms of a signomial coefficient vector) that rules out v.
    #
    # Use this function to improve a bound in any problem (constrained or unconstrained)
    # where the gap between SAGE and nonnegativity is viewed as the likely source of error.
    #
    # When using this function with a constrained problem, "v" and "alpha" should be associated
    # with the constraint "Lagrangian is SAGE".
    from sageopt.relaxations.sage_sigs import primal_sage_cone
    from sageopt.coniclifts.operators.norms import vector2norm
    c_x = cl.Variable(shape=(alpha.shape[0],))
    s_x = Signomial(alpha, c_x)
    modulator = Signomial(alpha, np.ones(shape=(alpha.shape[0],))) ** ell
    modded_s_x = s_x * modulator
    constraints = [primal_sage_cone(modded_s_x, name='sig_is_nonneg', AbK=AbK),
                   vector2norm(c_x) <= np.linalg.norm(v)]
    objective_expr = c_x @ v
    prob = cl.Problem(cl.MIN, objective_expr, constraints)
    prob.solve(solver=solver, verbose=False)
    cl.clear_variable_indices()
    if prob.status == 'solved' and prob.value < 0:
        c_val = c_x.value()
        cut_sig = Signomial(alpha, c_val)
        return cut_sig
    else:
        return None


def constrained_cut(v, alpha, gts, eqs, ell=1, solver='MOSEK'):
    #
    # Would be nice to infer a valid inequality that strengthens the Lagrange dual
    # (not just the SAGE relaxation of the Lagrange dual). Right now this function
    # only strengthens a given SAGE relaxation, and makes it at-most-as-strong-as
    # the Lagrange dual.
    #
    #
    from sageopt.relaxations.sage_sigs import primal_sage_cone
    from sageopt.coniclifts.operators.norms import vector2norm
    c_x0 = cl.Variable(shape=(alpha.shape[0],))
    s_x0 = Signomial(alpha, c_x0)
    modulator = Signomial(alpha, np.ones(shape=(alpha.shape[0],))) ** ell
    modded_s_x0 = s_x0 * modulator
    constraints = [primal_sage_cone(modded_s_x0, name='sig_is_nonneg', AbK=None)]
    c_x = c_x0
    if len(gts) > 0:
        lambda_vars = cl.Variable(shape=(len(gts),))
        constraints.append(lambda_vars >= 0)
        G = np.hstack([sym_corr.relative_coeff_vector(g, alpha).reshape((-1, 1)) for g in gts])
        c_g = G @ lambda_vars
        c_x = c_x + c_g
    if len(eqs) > 0:
        mu_vars = cl.Variable(shape=(len(eqs),))
        H = np.hstack([sym_corr.relative_coeff_vector(h, alpha).reshape((-1, 1)) for h in eqs])
        c_h = H @ mu_vars
        c_x = c_x + c_h
    constraints.append(vector2norm(c_x) <= np.linalg.norm(v))
    objective_expr = c_x @ v
    prob = cl.Problem(cl.MIN, objective_expr, constraints)
    prob.solve(solver=solver, verbose=False)
    cl.clear_variable_indices()
    if prob.status == 'solved' and prob.value < 0:
        c_val = c_x.value()
        cut_sig = Signomial(alpha, c_val)
        return cut_sig
    else:
        return None
