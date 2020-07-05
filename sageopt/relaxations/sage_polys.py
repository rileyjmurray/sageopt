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
from sageopt.symbolic.polynomials import Polynomial, PolyDomain
from sageopt.symbolic.signomials import Signomial
from sageopt.relaxations import sage_sigs
from sageopt.relaxations.sage_sigs import _check_kwargs
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


def poly_relaxation(f, X=None, form='dual', **kwargs):
    """
    Construct a coniclifts Problem instance for producing a lower bound on

    .. math::

        \min\{ f(x) \,:\, x \\in X \}

    where :math:`X=R^{\\texttt{f.n}}` by default.

    Parameters
    ----------
    f : Polynomial
        The objective function to be minimized.
    X : PolyDomain or None
        If ``X`` is None, then we produce a bound on ``f`` over :math:`R^{\\texttt{f.n}}`.
    form : str
        Either ``form='primal'`` or ``form='dual'``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    Notes
    -----

    This function also accepts the following keyword arguments:

    poly_ell : int
        Controls the complexity of a polynomial modulating function. Must be nonnegative.

    sigrep_ell : int
        Controls the complexity of the SAGE certificate applied to the Lagrangian's signomial representative.

    The dual formulation does not allow that both ``poly_ell`` and ``sigrep_ell`` are greater than
    zero (such functionality is not implemented at this time).

    Sageopt does not currently implement solution recovery for dual-form relaxations generated
    by this function. If you want to recover solutions, call ``poly_constrained_relaxation``
    with empty lists ``gts=[]`` and  ``eqs=[]``.
    """
    _check_kwargs(kwargs, allowed={'poly_ell', 'sigrep_ell'})
    poly_ell = kwargs['poly_ell'] if 'poly_ell' in kwargs else 0
    sigrep_ell = kwargs['sigrep_ell'] if 'sigrep_ell' in kwargs else 0
    form = form.lower()
    if form[0] == 'd':
        prob = poly_dual(f, poly_ell, sigrep_ell, X)
    elif form[0] == 'p':
        prob = poly_primal(f, poly_ell, sigrep_ell, X)
    else:  # pragma: no cover
        raise RuntimeError('Unrecognized form: ' + form + '.')
    return prob


def poly_dual(f, poly_ell=0, sigrep_ell=0, X=None):
    if poly_ell == 0:
        sr, _ = f.sig_rep
        prob = sage_sigs.sig_dual(sr, sigrep_ell, X=X)
        cl.clear_variable_indices()
        return prob
    elif sigrep_ell == 0:
        modulator = f.standard_multiplier() ** poly_ell
        gamma = cl.Variable()
        lagrangian = (f - gamma) * modulator
        v = cl.Variable(shape=(lagrangian.m, 1), name='v')
        con_base_name = v.name + ' domain'
        constraints = relative_dual_sage_poly_cone(lagrangian, v, con_base_name, log_AbK=X)
        a = sym_corr.relative_coeff_vector(modulator, lagrangian.alpha)
        constraints.append(a.T @ v == 1)
        f_mod = Polynomial(f.alpha, f.c) * modulator
        obj_vec = sym_corr.relative_coeff_vector(f_mod, lagrangian.alpha)
        obj = obj_vec.T @ v
        prob = cl.Problem(cl.MIN, obj, constraints)
        cl.clear_variable_indices()
        return prob
    else:  # pragma: no cover
        raise NotImplementedError()


def poly_primal(f, poly_ell=0, sigrep_ell=0, X=None):
    if poly_ell == 0:
        sr, _ = f.sig_rep
        prob = sage_sigs.sig_primal(sr, sigrep_ell, X=X)
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
            con = sage_sigs.primal_sage_cone(sig_under_test, con_name, X=X)
            constraints = [con] + cons
        else:
            con_name = 'Lagrangian sage poly'
            constraints = primal_sage_poly_cone(lagrangian, con_name, log_AbK=X)
        obj = gamma
        prob = cl.Problem(cl.MAX, obj, constraints)
        cl.clear_variable_indices()
        return prob


def sage_feasibility(f, X=None):
    """
    Constructs a coniclifts maximization Problem which is feasible iff ``f`` admits an X-SAGE decomposition.

    Parameters
    ----------
    f : Polynomial
        We want to test if this function admits an X-SAGE decomposition.
    X : PolyDomain or None
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts maximization Problem. If ``f`` admits an X-SAGE decomposition, then we should have
        ``prob.value > -np.inf``, once ``prob.solve()`` has been called.
    """
    sr, cons = f.sig_rep
    prob = sage_sigs.sage_feasibility(sr, X=X, additional_cons=cons)
    cl.clear_variable_indices()
    return prob


def sage_multiplier_search(f, level=1, X=None):
    """
    Constructs a coniclifts maximization Problem which is feasible if ``f`` can be certified as nonnegative
    over ``X``, by using an appropriate X-SAGE modulating function.

    Parameters
    ----------
    f : Polynomial
        We want to test if ``f`` is nonnegative over ``X``.
    level : int
        Controls the complexity of the X-SAGE modulating function. Must be a positive integer.
    X : PolyDomain or None
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    Notes
    -----
    This function provides an alternative to moving up the reference SAGE hierarchy, for the
    goal of certifying nonnegativity of a polynomial ``f`` over some set ``X`` where ``|X|``
    is log-convex. In general, the approach is to introduce a polynomial

        ``mult = Polynomial(alpha_hat, c_tilde)``

    where the rows of alpha_hat are all "level"-wise sums of rows from ``f.alpha``, and ``c_tilde``
    is a coniclifts Variable defining a nonzero SAGE polynomial. Then we can check if
    ``f_mod = f * mult`` is SAGE for any choice of ``c_tilde``.
    """
    constraints = []
    # Make the multiplier polynomial (and require that it be SAGE)
    mult_alpha = hierarchy_e_k([f], k=level)
    c_tilde = cl.Variable(shape=(mult_alpha.shape[0],), name='c_tilde')
    mult = Polynomial(mult_alpha, c_tilde)
    temp_cons = primal_sage_poly_cone(mult, name=(c_tilde.name + ' domain'), log_AbK=X)
    constraints += temp_cons
    constraints.append(cl.sum(c_tilde) >= 1)
    # Make "f_mod := f * mult", and require that it be SAGE.
    f_mod = mult * f
    temp_cons = primal_sage_poly_cone(f_mod, name='f_mod sage poly', log_AbK=X)
    constraints += temp_cons
    # noinspection PyTypeChecker
    prob = cl.Problem(cl.MAX, 0, constraints)
    cl.clear_variable_indices()
    return prob


def poly_constrained_relaxation(f, gts, eqs, X=None, form='dual', **kwargs):
    """
    Construct a coniclifts Problem representing a SAGE relaxation for the polynomial optimization
    problem

    .. math::

        \\begin{align*}
          \min\{ f(x) :~& g(x) \geq 0 \\text{ for } g \\in \\text{gts}, \\\\
                       & g(x) = 0  \\text{ for } g \\in \\text{eqs}, \\\\
                       & \\text{and } x \\in X \}
        \\end{align*}

    where :math:`X = R^{\\texttt{f.n}}` by default. The optimal value of this relaxation will
    produce a lower bound on the minimization problem described above. When ``form='dual'``,
    a solution to this  relaxation can be used to help recover optimal solutions to the problem
    described above.

    Parameters
    ----------
    f : Polynomial
        The objective to be minimized.
    gts : list of Polynomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Polynomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    X : PolyDomain
        If ``X`` is None, then we produce a bound on ``f`` subject only to the constraints in
        ``gts`` and ``eqs``.
    form : str
        Either ``form='primal'`` or ``form='dual'``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    Notes
    -----

    This function also accepts the following keyword arguments:

    p : int
        Controls the complexity of Lagrange multipliers on explicit polynomial constraints ``gts`` and ``eqs``.
        Defaults to ``p=0``, which corresponds to scalar Lagrange multipliers.
    q : int
        The lists ``gts`` and ``eqs`` are replaced by lists of polynomials formed by all products of ``<= q``
        elements from ``gts`` and ``eqs`` respectively. Defaults to ``q = 1``.
    ell : int
        Controls the strength of the SAGE proof system, as applied to the Lagrangian. Defaults to
        ``ell=0``, which means the primal Lagrangian must be an X-SAGE polynomial.
    slacks : bool
        For dual relaxations, determines if constraints "``mat @ vec`` in dual SAGE cone" is
        represented by "``mat @ vec == temp``, ``temp`` in dual SAGE cone". Defaults to False.
    """
    _check_kwargs(kwargs, allowed={'p', 'q', 'ell', 'slacks'})
    form = form.lower()
    p = kwargs['p'] if 'p' in kwargs else 0
    q = kwargs['q'] if 'q' in kwargs else 1
    ell = kwargs['ell'] if 'ell' in kwargs else 0
    if form[0] == 'd':
        prob = poly_constrained_dual(f, gts, eqs, p, q, ell, X)
    elif form[0] == 'p':
        prob = poly_constrained_primal(f, gts, eqs, p, q, ell, X)
    else:  # pragma: no cover
        raise RuntimeError('Unrecognized form: ' + form + '.')
    return prob


def poly_constrained_primal(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the primal SAGE-(p, q, ell) relaxation for the polynomial optimization problem

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where :math:`X = R^{\\texttt{f.n}}` by default.
    """
    lagrangian, ineq_lag_mults, _, gamma = make_poly_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian}
    if ell > 0:
        alpha_E_q = hierarchy_e_k([f] + list(gts) + list(eqs), k=1)
        modulator = Polynomial(2 * alpha_E_q, np.ones(alpha_E_q.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        metadata['modulator'] = modulator
    # The Lagrangian (after possible multiplication, as above) must be a SAGE polynomial.
    con_name = 'Lagrangian sage poly'
    constrs = primal_sage_poly_cone(lagrangian, con_name, log_AbK=X)
    #  Lagrange multipliers (for inequality constraints) must be SAGE polynomials.
    for s_h, _ in ineq_lag_mults:
        con_name = str(s_h) + ' domain'
        cons = primal_sage_poly_cone(s_h, con_name, log_AbK=X)
        constrs += cons
    # Construct the coniclifts problem.
    prob = cl.Problem(cl.MAX, gamma, constrs)
    prob.metadata = metadata
    cl.clear_variable_indices()
    return prob


def poly_constrained_dual(f, gts, eqs, p=0, q=1, ell=0, X=None, slacks=False):
    """
    Construct the dual SAGE-(p, q, ell) relaxation for the polynomial optimization problem

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where :math:`X = R^{\\texttt{f.n}}` by default.
    """
    lagrangian, ineq_lag_mults, eq_lag_mults, _ = make_poly_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'f': f, 'gts': gts, 'eqs': eqs, 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f, f.upcast_to_polynomial(1)] + gts + eqs, k=1)
        modulator = Polynomial(2 * alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        f = f * modulator
    else:
        modulator = f.upcast_to_polynomial(1)
    metadata['modulator'] = modulator
    # In primal form, the Lagrangian is constrained to be a SAGE polynomial.
    # Introduce a dual variable "v" for this constraint.
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    metadata['v_poly'] = v
    constraints = relative_dual_sage_poly_cone(lagrangian, v, 'Lagrangian', log_AbK=X)
    for s_g, g in ineq_lag_mults:
        # These generalized Lagrange multipliers "s_g" are SAGE polynomials.
        # For each such multiplier, introduce an appropriate dual variable "v_g", along
        # with constraints over that dual variable.
        g_m = g * modulator
        c_g = sym_corr.moment_reduction_array(s_g, g_m, lagrangian)
        name_base = 'v_' + str(g)
        if slacks:
            v_g = cl.Variable(name=name_base, shape=(s_g.m, 1))
            con = c_g @ v == v_g
            con.name += str(g) + ' >= 0'
            constraints.append(con)
        else:
            v_g = c_g @ v
        constraints += relative_dual_sage_poly_cone(s_g, v_g,
                                                    name_base=(name_base + ' domain'), log_AbK=X)
    for z_g, g in eq_lag_mults:
        # These generalized Lagrange multipliers "z_g" are arbitrary polynomials.
        # They dualize to homogeneous equality constraints.
        g_m = g * modulator
        c_g = sym_corr.moment_reduction_array(z_g, g_m, lagrangian)
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
    prob.metadata = metadata
    cl.clear_variable_indices()
    return prob


def make_poly_lagrangian(f, gts, eqs, p, q):
    """
    Given a problem

    .. math::

        \\begin{align*}
          \min\{ f(x) :~& g(x) \geq 0 \\text{ for } g \\in \\text{gts}, \\\\
                       & g(x) = 0  \\text{ for } g \\in \\text{eqs}, \\\\
                       & \\text{and } x \\in X \}
        \\end{align*}

    construct the q-fold constraints ``q-gts`` and ``q-eqs``, by taking all products
    of ``<= q`` elements from ``gts`` and ``eqs`` respectively. Then form the Lagrangian

    .. math::

        L = f - \\gamma
            - \sum_{g \, \\in  \, \\text{q-gts}} s_g \cdot g
            - \sum_{g \, \\in  \, \\text{q-eqs}} z_g \cdot g

    where :math:`\\gamma` is a coniclifts Variable of dimension 1, and the coefficients
    on Polynomials  :math:`s_g` and :math:`z_g` are coniclifts Variables of a dimension
    determined by ``p``.

    Parameters
    ----------
    f : Polynomial
        The objective in a desired minimization problem.
    gts : list of Polynomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Polynomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    p : int
        Controls the complexity of ``s_g`` and ``z_g``.
    q : int
        The number of folds of constraints ``gts`` and ``eqs``.

    Returns
    -------

    L : Polynomial
        ``L.c`` is an affine expression of coniclifts Variables.

    ineq_dual_polys : a list of pairs of Polynomials.
        If the pair ``(s_g, g)`` is in this list, then ``s_g`` is a generalized Lagrange multiplier
        to the constraint ``g(x) >= 0``.

    eq_dual_polys : a list of pairs of Polynomials.
        If the pair ``(z_g, g)`` is in this list, then ``z_g`` is a generalized Lagrange multiplier to the
        constraint ``g(x) == 0``.

    gamma : coniclifts.Variable.
        In primal-form SAGE relaxations, we want to maximize ``gamma``. In dual form SAGE relaxations,
        ``gamma`` induces a normalizing equality constraint.

    Notes
    -----
    The Lagrange multipliers ``s_g`` and ``z_g`` share a common matrix of exponent vectors,
    which we call ``alpha_hat``.

    When ``p = 0``, ``alpha_hat`` consists of a single row, of all zeros. In this case,
    ``s_g`` and ``z_g`` are constant Polynomials, and the coefficient vectors ``s_g.c``
    and ``z_g.c`` are effectively scalars. When ``p > 0``, the rows of ``alpha_hat`` are
    *initially* set set to all ``p``-wise sums  of exponent vectors appearing in either ``f``,
    or some ``g in gts``,  or some ``g in eqs``. Then we replace ::

        alpha_hat = np.vstack([2 * alpha_hat, alpha_hat])
        alpha_multiplier = np.unique(alpha_hat, axis=0)

    This has the effect of improving performance for problems where ``alpha_hat`` would otherwise
    contain very few rows in the even integer lattice.
    """
    folded_gt = con_gen.up_to_q_fold_cons(gts, q)
    gamma = cl.Variable(name='gamma')
    L = f - gamma
    alpha_E_p = hierarchy_e_k([f, f.upcast_to_polynomial(1)] + gts + eqs, k=p)
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


def infer_domain(f, gts, eqs, check_feas=True):
    """
    Identify a subset of the constraints in ``gts`` and ``eqs`` which can be incorporated into
    conditional SAGE relaxations for polynomials. Construct a PolyDomain object from the inferred
    constraints.

    Parameters
    ----------
    f : Polynomial
        The objective in a desired optimization problem. This parameter is only used to determine
        the dimension of the set defined by constraints in ``gts`` and ``eqs``.
    gts : list of Polynomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Polynomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    check_feas : bool
        Indicates whether or not to verify that the returned PolyDomain is nonempty.

    Returns
    -------
    X : PolyDomain or None

    """
    # GP-representable inequality constraints (recast as "Signomial >= 0")
    gp_gts = con_gen.valid_gp_representable_poly_inequalities(gts)
    gp_gts_sigreps = [Signomial(g.alpha, g.c) for g in gp_gts]
    gp_gts_sigreps = con_gen.valid_posynomial_inequalities(gp_gts_sigreps)
    #   ^ That second call is to convexify the signomials.
    # GP-representable equality constraints (recast as "Signomial == 0")
    gp_eqs = con_gen.valid_gp_representable_poly_eqs(eqs)
    gp_eqs_sigreps = [Signomial(g.alpha, g.c) for g in gp_eqs]
    gp_eqs_sigreps = con_gen.valid_monomial_equations(gp_eqs_sigreps)
    #  ^ That second call is to make sure the nonconstant term has
    #    a particular sign (specifically, a negative sign).
    clcons = con_gen.clcons_from_standard_gprep(f.n, gp_gts_sigreps, gp_eqs_sigreps)
    if len(clcons) > 0:
        polydom = PolyDomain(f.n, logspace_cons=clcons,
                             gts=gp_gts, eqs=gp_eqs,
                             check_feas=check_feas)
        return polydom
    else:
        return None


def hierarchy_e_k(polys, k):
    alphas = [s.alpha for s in polys]
    alpha = np.vstack(alphas)
    alpha = np.unique(alpha, axis=0)
    c = np.ones(shape=(alpha.shape[0],))
    s = Polynomial(alpha, c)
    s = s ** k
    return s.alpha

