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
from sageopt import coniclifts as cl
from sageopt.symbolic.polynomials import Polynomial
from sageopt.symbolic.signomials import Signomial
from sageopt.relaxations import sage_sigs
from sageopt.relaxations import constraint_generators as con_gen
from sageopt.relaxations import symbolic_correspondences as sym_corr


def primal_sage_poly_cone(poly, name, log_AbK):
    poly_sr, poly_sr_cons = poly.sig_rep
    con = sage_sigs.primal_sage_cone(poly_sr, name, log_AbK)
    constrs = [con] + poly_sr_cons
    return constrs


def relative_dual_sage_poly_cone(primal_poly, dual_var, name_base, log_AbK):
    """
    :param log_AbK:
    :param primal_poly: a Polynomial
    :param dual_var: a coniclifts Variable with y.shape == (p.m, 1).
    :param name_base:

    :return: coniclifts Constraints over y (and additional auxilliary variables, as
    necessary) so that y defines a dual variable to the constraint that "p is a SAGE polynomial."
    """
    sr, sr_cons = primal_poly.sig_rep
    evens = [i for i, row in enumerate(sr.alpha) if np.all(row % 2 == 0)]
    if len(evens) < sr.m:
        is_even = np.zeros(shape=(sr.m,), dtype=bool)
        is_even[evens] = True
        aux_v = cl.Variable(shape=(sr.m, 1), name='aux_v_{' + name_base + ' sage poly dual}')
        constrs = [sage_sigs.relative_dual_sage_cone(sr, aux_v, name_base + ' sigrep sage dual', log_AbK),
                   aux_v[is_even] == dual_var[is_even],
                   -aux_v[~is_even] <= dual_var[~is_even], dual_var[~is_even] <= aux_v[~is_even]]
    else:
        constrs = [sage_sigs.relative_dual_sage_cone(sr, dual_var, name_base + ' sigrep sage dual', log_AbK)]
    return constrs


def poly_dual(f, poly_ell=0, sigrep_ell=0, X=None):
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    if poly_ell == 0:
        sr, cons = f.sig_rep
        if len(cons) > 0:
            msg = '\n\nThe provided Polynomial has nonconstant coefficients.\n'
            msg += 'The most likely cause is that a mistake has been made in setting up '
            msg += 'the data for this function.\n Raising a RuntimeError.\n'
            raise RuntimeError(msg)
        log_X = {'AbK':  X['log_AbK'],
                 'gts': [log_domain_converter(g) for g in X['gts']],
                 'eqs': [log_domain_converter(g) for g in X['eqs']]}
        prob = sage_sigs.sig_dual(sr, sigrep_ell, X=log_X)
        cl.clear_variable_indices()
        return prob
    elif sigrep_ell == 0:
        modulator = f.standard_multiplier() ** poly_ell
        gamma = cl.Variable()
        lagrangian = (f - gamma) * modulator
        v = cl.Variable(shape=(lagrangian.m, 1), name='v')
        con_base_name = v.name + ' domain'
        constraints = relative_dual_sage_poly_cone(lagrangian, v, con_base_name, log_AbK=X['log_AbK'])
        a = sym_corr.relative_coeff_vector(modulator, lagrangian.alpha)
        constraints.append(a.T @ v == 1)
        f_mod = Polynomial(f.alpha_c) * modulator
        obj_vec = sym_corr.relative_coeff_vector(f_mod, lagrangian.alpha)
        obj = obj_vec.T @ v
        prob = cl.Problem(cl.MIN, obj, constraints)
        cl.clear_variable_indices()
        return prob
    else:
        raise NotImplementedError()


def poly_primal(f, poly_ell=0, sigrep_ell=0, X=None):
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    if poly_ell == 0:
        sr, cons = f.sig_rep
        if len(cons) > 0:
            msg = '\n\nThe provided Polynomial has nonconstant coefficients.\n'
            msg += 'The most likely cause is that a mistake has been made in setting up '
            msg += 'the data for this function.\n Raising a RuntimeError.\n'
            raise RuntimeError(msg)
        log_X = {'AbK':  X['log_AbK'],
                 'gts': [log_domain_converter(g) for g in X['gts']],
                 'eqs': [log_domain_converter(g) for g in X['eqs']]}
        prob = sage_sigs.sig_primal(sr, sigrep_ell, X=log_X, additional_cons=cons)
        cl.clear_variable_indices()
        return prob
    else:
        poly_modulator = f.standard_multiplier() ** poly_ell
        gamma = cl.Variable(shape=(), name='gamma')
        lagrangian = (f - gamma) * poly_modulator
        if sigrep_ell > 0:
            sr, cons = lagrangian.sig_rep
            sig_modulator = Signomial(sr.alpha, np.ones(shape=(sr.m,))) ** sigrep_ell
            sig_under_test = sr * sig_modulator
            con_name = 'Lagrangian modulated sigrep sage'
            con = sage_sigs.primal_sage_cone(sig_under_test, con_name, AbK=X['log_AbK'])
            constraints = [con] + cons
        else:
            con_name = 'Lagrangian sage poly'
            constraints = primal_sage_poly_cone(lagrangian, con_name, log_AbK=X['log_AbK'])
        obj = gamma
        prob = cl.Problem(cl.MAX, obj, constraints)
        cl.clear_variable_indices()
        return prob


def sage_feasibility(f, X=None):
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    log_X = {'AbK': X['log_AbK'],
             'gts': [log_domain_converter(g) for g in X['gts']],
             'eqs': [log_domain_converter(g) for g in X['eqs']]}
    sr, cons = f.sig_rep
    prob = sage_sigs.sage_feasibility(sr, X=log_X, additional_cons=cons)
    cl.clear_variable_indices()
    return prob


def sage_multiplier_search(f, level=1, X=None):
    """
    Suppose we have a nonnegative polynomial f that is not SAGE. Do we have an alternative
    to proving that f is nonnegative other than moving up the usual SAGE hierarchy?
    Indeed we do. We can define a multiplier

        mult = Polynomial(alpha_hat, c_tilde)

    where the rows of alpha_hat are all "level"-wise sums of rows from f.alpha, and c_tilde
    is a coniclifts Variable defining a nonzero SAGE polynomial. Then we can check if
    f_mod := f * mult is SAGE for any choice of c_tilde.

    :param f: a Polynomial object
    :param level: a nonnegative integer
    :param X: None, or a dictionary with three keys 'log_AbK', 'gts', 'eqs' such as that
    generated by the function "conditional_sage_data(...)".

    :return: a coniclifts maximization Problem that is feasible iff f * mult is SAGE
    for some SAGE multiplier Polynomial "mult".
    """
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    constraints = []
    # Make the multiplier polynomial (and require that it be SAGE)
    mult_alpha = hierarchy_e_k([f], k=level)
    c_tilde = cl.Variable(shape=(mult_alpha.shape[0],), name='c_tilde')
    mult = Polynomial(mult_alpha, c_tilde)
    temp_cons = primal_sage_poly_cone(mult, name=(c_tilde.name + ' domain'), log_AbK=X['log_AbK'])
    constraints += temp_cons
    constraints.append(cl.sum(c_tilde) >= 1)
    # Make "f_mod := f * mult", and require that it be SAGE.
    f_mod = mult * f
    temp_cons = primal_sage_poly_cone(f_mod, name='f_mod sage poly', log_AbK=X['log_AbK'])
    constraints += temp_cons
    # noinspection PyTypeChecker
    prob = cl.Problem(cl.MAX, 0, constraints)
    cl.clear_variable_indices()
    return prob


def poly_constrained_primal(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the primal SAGE-(p, q, ell) relaxation for the polynomial optimization problem

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and log(|x|) in X }

    where X = [R \\union {-infty}]^(f.n) by default.

    :param f: a Polynomial.
    :param gts: a list of Polynomials.
    :param eqs: a list of Polynomials.
    :param p: a nonnegative integer.
        Controls the complexity of Lagrange multipliers. p=0 corresponds to scalars.
    :param q: a positive integer.
        The number of folds applied to the constraints "gts" and "eqs". p=1 means "leave gts and eqs as-is."
    :param ell: a nonnegative integer.
        Controls the complexity of any modulator applied to the Lagrangian. ell=0 means that
        the Lagrangian must be SAGE. ell=1 means "tilde_L := modulator * Lagrangian" must be SAGE.
    :param X: None, or a dictionary with three keys 'log_AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".

    :return: The primal form SAGE-(p, q, ell) relaxation for the given polynomial optimization problem.
    """
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, _, gamma = make_poly_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian}
    if ell > 0:
        alpha_E_q = hierarchy_e_k([f] + list(gts) + list(eqs), k=1)
        modulator = Polynomial(2 * alpha_E_q, np.ones(alpha_E_q.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        metadata['modulator'] = modulator
    # The Lagrangian (after possible multiplication, as above) must be a SAGE polynomial.
    con_name = 'Lagrangian sage poly'
    constrs = primal_sage_poly_cone(lagrangian, con_name, log_AbK=X['log_AbK'])
    #  Lagrange multipliers (for inequality constraints) must be SAGE polynomials.
    for s_h, _ in ineq_lag_mults:
        con_name = str(s_h) + ' domain'
        cons = primal_sage_poly_cone(s_h, con_name, log_AbK=X['log_AbK'])
        constrs += cons
    # Construct the coniclifts problem.
    prob = cl.Problem(cl.MAX, gamma, constrs)
    prob.associated_data = metadata
    cl.clear_variable_indices()
    return prob


def poly_constrained_dual(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the dual SAGE-(p, q, ell) relaxation for the polynomial optimization problem

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and log(|x|) in X }

    where X = [R \\union {-infty}]^(f.n) by default.

    :param f: a Polynomial.
    :param gts: a list of Polynomials.
    :param eqs: a list of Polynomials.
    :param p: a nonnegative integer.
        Controls the complexity of Lagrange multipliers. p=0 corresponds to scalars.
    :param q: a positive integer.
        The number of folds applied to the constraints "gts" and "eqs". p=1 means "leave gts and eqs as-is."
    :param ell: a nonnegative integer.
        Controls the complexity of any modulator applied to the Lagrangian. ell=0 means that
        the Lagrangian must be SAGE. ell=1 means "tilde_L := modulator * Lagrangian" must be SAGE.
    :param X: None, or a dictionary with three keys 'log_AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".

    :return: The dual SAGE-(p, q, ell) relaxation for the given polynomial optimization problem.
    """
    if X is None:
        X = {'log_AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, eq_lag_mults, _ = make_poly_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'f': f, 'gts': gts, 'eqs': eqs, 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f] + list(gts) + list(eqs), k=1)
        modulator = Polynomial(2 * alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        f = f * modulator
    else:
        modulator = Polynomial({(0,) * f.n: 1})
    metadata['modulator'] = modulator
    # In primal form, the Lagrangian is constrained to be a SAGE polynomial.
    # Introduce a dual variable "v" for this constraint.
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    metadata['v_poly'] = v
    constraints = relative_dual_sage_poly_cone(lagrangian, v, 'Lagrangian', log_AbK=X['log_AbK'])
    for s_g, g in ineq_lag_mults:
        # These generalized Lagrange multipliers "s_g" are SAGE polynomials.
        # For each such multiplier, introduce an appropriate dual variable "v_g", along
        # with constraints over that dual variable.
        v_g = cl.Variable(name='v_' + str(g), shape=(s_g.m, 1))
        constraints += relative_dual_sage_poly_cone(s_g, v_g, name_base=(v_g.name + ' domain'), log_AbK=X['log_AbK'])
        g = g * modulator
        c_g = sym_corr.moment_reduction_array(s_g, g, lagrangian)
        con = c_g @ v == v_g
        con.name += str(g) + ' >= 0'
        constraints.append(con)
    for z_g, g in eq_lag_mults:
        # These generalized Lagrange multipliers "z_g" are arbitrary polynomials.
        # They dualize to homogeneous equality constraints.
        g = g * modulator
        c_g = sym_corr.moment_reduction_array(z_g, g, lagrangian)
        con = c_g @ v == 0
        con.name += str(g) + ' == 0'
        constraints.append(con)
    # Equality constraint (for the Lagrangian to be bounded).
    a = sym_corr.relative_coeff_vector(modulator, lagrangian.alpha)
    constraints.append(a.T @ v == 1)
    # Define the dual objective function.
    obj_vec = sym_corr.relative_coeff_vector(f, lagrangian.alpha)
    obj = obj_vec.T @ v
    # Return the coniclifts Problem.
    prob = cl.Problem(cl.MIN, obj, constraints)
    prob.associated_data = metadata
    cl.clear_variable_indices()
    return prob


def make_poly_lagrangian(f, gts, eqs, p, q):
    """
    Given a problem

        inf_{x in X}{ f(x) : g(x) >= 0 for g in gts, g(x) == 0 for g in eqs },

    construct the q-fold constraints "folded_gts" and "folded_eqs," and the Lagrangian

        L = f - gamma - sum_{g in folded_gts} s_g * g - sum_{g in folded_eqs} z_g * g

    where gamma and the coefficients on Polynomials s_g / z_g are coniclifts Variables.

    The values returned by this function are used to construct constrained SAGE relaxations.
    The basic primal SAGE relaxation is obtained by maximizing gamma, subject to the constraint
    that L and each s_g are SAGE polynomials. The dual SAGE relaxation is obtained by symbolically
    applying conic duality to the primal.

    :param f: a Polynomial.
    :param gts: a list of Polynomials.
    :param eqs: a list of Polynomials.
    :param p: a nonnegative integer. Controls the complexity of s_g and z_g.
    :param q: a positive integer. The number of folds of constraints "gts" and "eqs".

    :return: L, ineq_dual_polys, eq_dual_polys, gamma.

        L : a Polynomial object with coefficients as affine expressions of coniclifts Variables.

        ineq_dual_polys : a list of pairs of Polynomial objects. If the pair (s1, s2) is in this list,
            then s1 is a generalized Lagrange multiplier to the constraint that s2(x) >= 0.

        eq_dual_polys : a list of pairs of Polynomial objects. If the pair (s1, s2) is in this list, then
            s1 is a generalized Lagrange multiplier to the constraint that s2(x) == 0.
            This return value is not accessed for primal-form SAGE relaxations.

        gamma : an unconstrained coniclifts Variable. This is the objective for primal-form SAGE relaxations,
            and induces a normalizing equality constraint in dual-form SAGE relaxations.
            This return value is not accessed for dual-form SAGE relaxations.
    """
    folded_gt = con_gen.up_to_q_fold_cons(gts, q)
    gamma = cl.Variable(name='gamma')
    L = f - gamma
    alpha_E_p = hierarchy_e_k([f] + list(gts) + list(eqs), k=p)
    alpha_multiplier = np.vstack([2 * alpha_E_p, alpha_E_p])
    alpha_multiplier = np.unique(alpha_multiplier, axis=0)
    ineq_dual_polys = []
    for g in folded_gt:
        s_g_coeff = cl.Variable(name='s_' + str(g), shape=(alpha_multiplier.shape[0],))
        s_g = Polynomial(alpha_multiplier, s_g_coeff)
        L -= s_g * g
        ineq_dual_polys.append((s_g, g))
    eq_dual_polys = []
    folded_eq = con_gen.up_to_q_fold_cons(eqs, q)
    for g in folded_eq:
        z_g_coeff = cl.Variable(name='z_' + str(g), shape=(alpha_multiplier.shape[0],))
        z_g = Polynomial(alpha_multiplier, z_g_coeff)
        L -= z_g * g
        eq_dual_polys.append((z_g, g))
    return L, ineq_dual_polys, eq_dual_polys, gamma


def conditional_sage_data(f, gts, eqs):
    """
    :param f: objective Polynomial
    :param gts: inequality constraint Polynomials
    :param eqs: equality constraint Polynomials

    :return: A dictionary X, keyed by three strings: 'log_AbK', 'gts', and 'eqs'.

    X['gts'] is a list of Polynomials so that every g in X['gts'] has an efficient
    convex representation for {log(|x|) : g(|x|) >= 0, |x| > 0}. (Where the vertical
    bars denote elementwise absolute value, and the logarithm is meant elementwise.)
    The intersection of all of these sets is contained within

            {log(|x|) : g(|x|) >= 0 for all g in gts, |x| > 0}.

    X['eqs'] is defined similarly, but for equality constraints.

    If both X['gts'] and X['eqs'] are empty, then X['log_AbK'] is None.

    Otherwise, X['log_AbK'] is a conic representation of the pointwise, elementwise
    log-absolute-values of the feasible sets cut out by X['gts'] and X['eqs'].

    The conic representation is a triple X['log_AbK'] = (A, b, K), where A is a SciPy
    sparse matrix, b is a numpy 1d array, and K is a list of Coniclifts Cone objects.
    The number of columns for "A" in X['AbK'] will always be at least f.n. If the number
    of columns is greater than f.n, then the first f.n columns of A correspond (in order!)
    to the log-absolute-values of variables over which f is  defined. Any remaining
    columns are auxiliary variables needed to represent X in coniclifts primitives.

    Notes:

        This function essentially defines the requirements for "X" which may be passed to
        conditional SAGE polynomial relaxations defined in this python module.

        It is possible for a user to properly define their own dict "X" without calling
        this function. The  only benefit to such an approach is that X['gts'] and X['eqs']
        don't need to be Polynomial objects. As long as X['gts'] and X['eqs'] are callable
        python functions and relate to X['log_AbK'] in the manner described above, then
        you should be able to pass that dict to SAGE relaxations defined in this module
        without trouble. Bear in mind that the functions in X['gts'] and X['eqs'] will
        only be passed elementwise-positive arguments.

    """
    # GP-representable inequality constraints (recast as "Signomial >= 0")
    gp_gts = con_gen.valid_gp_representable_poly_inequalities(gts)
    gp_gts_sigreps = [Signomial(g.alpha_c) for g in gp_gts]
    # GP-representable equality constraints (recast as "Signomial == 0")
    gp_eqs = con_gen.valid_gp_representable_poly_eqs(eqs)
    gp_eqs_sigreps = [Signomial(g.alpha_c) for g in gp_eqs]
    # Fall back on conditional SAGE data implementation for signomials
    dummy_f = Signomial({(0,) * f.n: 1})
    logX = sage_sigs.conditional_sage_data(dummy_f, gp_gts_sigreps, gp_eqs_sigreps)
    X = {'log_AbK': logX['AbK'], 'gts': gp_gts, 'eqs': gp_eqs}
    return X


def hierarchy_e_k(polys, k):
    alpha_tups = sum([list(s.alpha_c.keys()) for s in polys], [])
    alpha_tups = set(alpha_tups)
    s = Polynomial(dict([(a, 1.0) for a in alpha_tups]))
    s = s ** k
    return s.alpha


def log_domain_converter(f):
    fhat = lambda x: f(np.exp(x))
    return fhat
