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
from sageopt.symbolic.signomials import Signomial, SigDomain
from sageopt.relaxations import constraint_generators as con_gen
from sageopt.relaxations import symbolic_correspondences as sym_corr


def primal_sage_cone(sig, name, X, expcovers=None):
    con = cl.PrimalSageCone(sig.c, sig.alpha, X, name, covers=expcovers)
    return con


def relative_dual_sage_cone(primal_sig, dual_var, name, X, expcovers=None):
    con = cl.DualSageCone(dual_var, primal_sig.alpha, X, name, c=primal_sig.c, covers=expcovers)
    return con


def sig_relaxation(f, X=None, form='dual', **kwargs):
    """
    Construct a coniclifts Problem instance for producing a lower bound on

    .. math::

        f_X^{\\star} \doteq \min\{ f(x) \,:\, x \\in X \}

    where X = :math:`\\mathbb{R}^{\\texttt{f.n}}` by default.

    When ``form='dual'``, a solution to this convex relaxation can be used to
    recover optimal solutions to the problem above. Refer to the Notes for keyword
    arguments accepted by this function.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    X : SigDomain
        If ``X`` is None, then we produce a bound on ``f`` over :math:`R^{\\texttt{f.n}}`.
    form : str
        Either ``form='primal'`` or ``form='dual'``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts Problem which represents the SAGE relaxation, with given parameters.
        The relaxation can be solved by calling ``prob.solve()``.

    Notes
    -----

    This function also accepts the following keyword arguments:

    ell : int
        The level of the reference SAGE hierarchy. Must be nonnegative.

    mod_supp : NumPy ndarray
        Only used when ``ell > 0``. If ``mod_supp`` is not None, then the rows of this
        array define the exponents of a positive definite modulating Signomial ``t`` in the reference SAGE hierarchy.
    """
    _check_kwargs(kwargs, allowed={'ell', 'mod_supp'})
    ell = kwargs['ell'] if 'ell' in kwargs else 0
    mod_supp = kwargs['mod_supp'] if 'mod_supp' in kwargs else None
    if form.lower()[0] == 'd':
        prob = sig_dual(f, ell, X, mod_supp)
    elif form.lower()[0] == 'p':
        prob = sig_primal(f, ell, X, mod_supp)
    else:
        raise RuntimeError('Unrecognized form: ' + form + '.')
    return prob


def sig_dual(f, ell=0, X=None, modulator_support=None):
    f = f.without_zeros()
    # Signomial definitions (for the objective).
    lagrangian = f - cl.Variable(name='gamma')
    if modulator_support is None:
        modulator_support = lagrangian.alpha
    t_mul = Signomial(modulator_support, np.ones(modulator_support.shape[0])) ** ell
    metadata = {'f': f, 'lagrangian': lagrangian, 'modulator': t_mul, 'X': X}
    lagrangian = lagrangian * t_mul
    f_mod = f * t_mul
    # C_SAGE^STAR (v must belong to the set defined by these constraints).
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    con = relative_dual_sage_cone(lagrangian, v, name='Lagrangian SAGE dual constraint', X=X)
    constraints = [con]
    # Equality constraint (for the Lagrangian to be bounded).
    a = sym_corr.relative_coeff_vector(t_mul, lagrangian.alpha)
    a = a.reshape(a.size, 1)
    constraints.append(a.T @ v == 1)
    # Objective definition and problem creation.
    obj_vec = sym_corr.relative_coeff_vector(f_mod, lagrangian.alpha)
    obj = obj_vec.T @ v
    # Create coniclifts Problem
    prob = cl.Problem(cl.MIN, obj, constraints)
    prob.metadata = metadata
    cl.clear_variable_indices()
    return prob


def sig_primal(f, ell=0, X=None, modulator_support=None):
    f = f.without_zeros()
    gamma = cl.Variable(name='gamma')
    lagrangian = f - gamma
    if modulator_support is None:
        modulator_support = lagrangian.alpha
    t = Signomial(modulator_support, np.ones(modulator_support.shape[0]))
    s_mod = lagrangian * (t ** ell)
    con = primal_sage_cone(s_mod, name=str(s_mod), X=X)
    constraints = [con]
    obj = gamma.as_expr()
    prob = cl.Problem(cl.MAX, obj, constraints)
    cl.clear_variable_indices()
    return prob


def sage_feasibility(f, X=None, additional_cons=None):
    """
    Constructs a coniclifts maximization Problem which is feasible if and only if
    ``f`` admits an X-SAGE decomposition (:math:`X=R^{\\texttt{f.n}}` by default).

    Parameters
    ----------
    f : Signomial
        We want to test if this function admits an X-SAGE decomposition.
    X : SigDomain
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.
    additional_cons : :obj:`list` of :obj:`sageopt.coniclifts.Constraint`
        This is mostly used for SAGE polynomials. When provided, it should be a list of Constraints over
        coniclifts Variables appearing in ``f.c``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts maximization Problem. If ``f`` admits an X-SAGE decomposition, then we should have
        ``prob.value > -np.inf``, once ``prob.solve()`` has been called.
    """
    f = f.without_zeros()
    con = primal_sage_cone(f, name=str(f), X=X)
    constraints = [con]
    if additional_cons is not None:
        constraints += additional_cons
    prob = cl.Problem(cl.MAX, cl.Expression([0]), constraints)
    cl.clear_variable_indices()
    return prob


def sage_multiplier_search(f, level=1, X=None):
    """
    Constructs a coniclifts maximization Problem which is feasible if ``f`` can be certified as nonnegative
    over ``X``, by using an appropriate X-SAGE modulating function.

    Parameters
    ----------
    f : Signomial
        We want to test if ``f`` is nonnegative over ``X``.
    level : int
        Controls the complexity of the X-SAGE modulating function. Must be a positive integer.
    X : SigDomain
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.


    Returns
    -------
    prob : sageopt.coniclifts.Problem

    Notes
    -----
    This function provides an alternative to moving up the reference SAGE hierarchy, for the goal of certifying
    nonnegativity of a signomial ``f`` over some convex set ``X``.  In general, the approach is to introduce
    a signomial

        ``mult = Signomial(alpha_hat, c_tilde)``

    where the rows of ``alpha_hat`` are all ``level``-wise sums of rows from ``f.alpha``, and ``c_tilde``
    is a coniclifts Variable defining a nonzero X-SAGE function. Then we check if ``f_mod = f * mult``
    is X-SAGE for any choice of ``c_tilde``.
    """
    f = f.without_zeros()
    constraints = []
    mult_alpha = hierarchy_e_k([f, f.upcast_to_signomial(1)], k=level)
    c_tilde = cl.Variable(mult_alpha.shape[0], name='c_tilde')
    mult = Signomial(mult_alpha, c_tilde)
    constraints.append(cl.sum(c_tilde) >= 1)
    sig_under_test = mult * f
    con1 = primal_sage_cone(mult, name=str(mult), X=X)
    con2 = primal_sage_cone(sig_under_test, name=str(sig_under_test), X=X)
    constraints.append(con1)
    constraints.append(con2)
    prob = cl.Problem(cl.MAX, cl.Expression([0]), constraints)
    cl.clear_variable_indices()
    return prob


def sig_constrained_relaxation(f, gts, eqs, X=None, form='dual', **kwargs):
    """
    Construct a coniclifts Problem representing a SAGE relaxation for the signomial program

    .. math::

        \\begin{align*}
          \min\{ f(x) :~& g(x) \geq 0 \\text{ for } g \\in \\mathtt{gts}, \\\\
                       & g(x) = 0  \\text{ for } g \\in \\mathtt{eqs}, \\\\
                       & \\text{and } x \\in X \}
        \\end{align*}

    where X = :math:`R^{\\texttt{f.n}}` by default. When ``form='dual'``, a solution to this
    relaxation can be used  to help recover optimal solutions to the problem described above.
    Refer to the Notes for keyword arguments accepted by this function.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    gts : list of Signomial
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomial
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    X : SigDomain
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
        Controls the complexity of Lagrange multipliers on explicit signomial constraints ``gts`` and ``eqs``.
        Defaults to ``p=0``, which corresponds to scalar Lagrange multipliers.
    q : int
        The lists ``gts`` and ``eqs`` are replaced by lists of signomials formed by all products of ``<= q``
        elements from ``gts`` and ``eqs`` respectively. Defaults to ``q = 1``.
    ell : int
        Controls the strength of the SAGE proof system, as applied to the Lagrangian. Defaults to
        ``ell=0``, which means the primal Lagrangian must be an X-SAGE signomial.
    slacks : bool
        For dual relaxations, determines if constraints "``mat @ vec`` in dual SAGE cone" is
        represented by "``mat @ vec == temp``, ``temp`` in dual SAGE cone". Defaults to False.
    """
    _check_kwargs(kwargs, allowed={'p', 'q', 'ell', 'slacks'})
    p = kwargs['p'] if 'p' in kwargs else 0
    q = kwargs['q'] if 'q' in kwargs else 1
    ell = kwargs['ell'] if 'ell' in kwargs else 0
    slacks = kwargs['slacks'] if 'slacks' in kwargs else False

    if form.lower()[0] == 'd':
        prob = sig_constrained_dual(f, gts, eqs, p, q, ell, X, slacks)
    elif form.lower()[0] == 'p':
        prob = sig_constrained_primal(f, gts, eqs, p, q, ell, X)
    else:
        raise RuntimeError('Unrecognized form: ' + form + '.')
    return prob
    pass


def sig_constrained_primal(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the SAGE-(p, q, ell) primal problem for the signomial program

        min{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where X = :math:`R^{\\texttt{f.n}}` by default.
    """
    lagrangian, ineq_lag_mults, _, gamma = make_sig_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f, f.upcast_to_signomial(1)] + gts + eqs, k=1)
        modulator = Signomial(alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
    else:
        modulator = f.upcast_to_signomial(1)
    metadata['modulator'] = modulator
    # The Lagrangian (after possible multiplication, as above) must be a SAGE signomial.
    con = primal_sage_cone(lagrangian, name='Lagrangian is SAGE', X=X)
    constrs = [con]
    #  Lagrange multipliers (for inequality constraints) must be SAGE signomials.
    expcovers = None
    for i, (s_h, _) in enumerate(ineq_lag_mults):
        con_name = 'SAGE multiplier for signomial inequality # ' + str(i)
        con = primal_sage_cone(s_h, name=con_name, X=X, expcovers=expcovers)
        expcovers = con.ech.expcovers  # only * really * needed in first iteration, but keeps code flat.
        constrs.append(con)
    # Construct the coniclifts Problem.
    prob = cl.Problem(cl.MAX, gamma, constrs)
    prob.metadata = metadata
    cl.clear_variable_indices()
    return prob


def sig_constrained_dual(f, gts, eqs, p=0, q=1, ell=0, X=None, slacks=False):
    """
    Construct the SAGE-(p, q, ell) dual problem for the signomial program

        min{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where X = :math:`R^{\\texttt{f.n}}` by default.
    """
    lagrangian, ineq_lag_mults, eq_lag_mults, _ = make_sig_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'f': f, 'gts': gts, 'eqs': eqs, 'level': (p, q, ell), 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f, f.upcast_to_signomial(1)] + list(gts) + list(eqs), k=1)
        modulator = Signomial(alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        f = f * modulator
    else:
        modulator = f.upcast_to_signomial(1)
    metadata['modulator'] = modulator
    # In primal form, the Lagrangian is constrained to be a SAGE signomial.
    # Introduce a dual variable "v" for this constraint.
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    con = relative_dual_sage_cone(lagrangian, v, name='Lagrangian SAGE dual constraint', X=X)
    constraints = [con]
    expcovers = None
    for i, (s_h, h) in enumerate(ineq_lag_mults):
        # These generalized Lagrange multipliers "s_h" are SAGE signomials.
        # For each such multiplier, introduce an appropriate dual variable "v_h", along
        # with constraints over that dual variable.
        h_m = h * modulator
        c_h = sym_corr.moment_reduction_array(s_h, h_m, lagrangian)
        if slacks:
            v_h = cl.Variable(name='v_' + str(h), shape=(s_h.m, 1))
            constraints.append(c_h @ v == v_h)
        else:
            v_h = c_h @ v
        con_name = 'SAGE dual for signomial inequality # ' + str(i)
        con = relative_dual_sage_cone(s_h, v_h, name=con_name, X=X, expcovers=expcovers)
        expcovers = con.ech.expcovers  # only * really * needed in first iteration, but keeps code flat.
        constraints.append(con)
    for s_h, h in eq_lag_mults:
        # These generalized Lagrange multipliers "s_h" are arbitrary signomials.
        # They dualize to homogeneous equality constraints.
        h = h * modulator
        c_h = sym_corr.moment_reduction_array(s_h, h, lagrangian)
        constraints.append(c_h @ v == 0)
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


def make_sig_lagrangian(f, gts, eqs, p, q):
    """
    Given a problem

    .. math::

        \\begin{align*}
          \min\{ f(x) :~& g(x) \geq 0 \\text{ for } g \\in \\mathtt{gts}, \\\\
                       & g(x) = 0  \\text{ for } g \\in \\mathtt{eqs}, \\\\
                       & \\text{and } x \\in X \}
        \\end{align*}

    construct the q-fold constraints ``q-gts`` and ``q-eqs``, by taking all products
    of ``<= q`` elements from ``gts`` and ``eqs`` respectively. Then form the Lagrangian

    .. math::

        L = f - \\gamma
            - \sum_{g \, \\in  \, \\mathtt{q-gts}} s_g \cdot g
            - \sum_{g \, \\in  \, \\mathtt{q-eqs}} z_g \cdot g

    where :math:`\\gamma` is a coniclifts Variable of dimension 1, and the coefficients
    on Signomials  :math:`s_g` and :math:`z_g` are coniclifts Variables of a dimension
    determined by ``p``.

    Parameters
    ----------
    f : Signomial
        The objective in a desired minimization problem.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    p : int
        Controls the complexity of ``s_g`` and ``z_g``.
    q : int
        The number of folds of constraints ``gts`` and ``eqs``.

    Returns
    -------
    L : Signomial
        ``L.c`` is an affine expression of coniclifts Variables.

    ineq_dual_sigs : a list of pairs of Signomial objects.
        If the pair ``(s_g, g)`` is in this list, then ``s_g`` is a generalized Lagrange multiplier
        to the constraint ``g(x) >= 0``.

    eq_dual_sigs : a list of pairs of Signomial objects.
        If the pair ``(z_g, g)`` is in this list, then ``z_g`` is a generalized Lagrange multiplier to the
        constraint ``g(x) == 0``.

    gamma : coniclifts.Variable.
        In primal-form SAGE relaxations, we want to maximize ``gamma``. In dual form SAGE relaxations,
        ``gamma`` induces an equality constraint.

    Notes
    -----
    The Lagrange multipliers ``s_g`` and ``z_g`` share a common matrix of exponent vectors,
    which we call ``alpha_hat``.

    When ``p = 0``, ``alpha_hat`` consists of a single row, of all zeros. In this case,
    ``s_g`` and ``z_g`` are constant Signomials, and the coefficient vectors ``s_g.c``
    and ``z_g.c`` are effectively scalars. When ``p > 0``, the rows of ``alpha_hat`` are
    set to all ``p``-wise sums  of exponent vectors appearing in either ``f``, or some
    ``g in gts``,  or some ``g in eqs``.
    """
    folded_gt = con_gen.up_to_q_fold_cons(gts, q)
    gamma = cl.Variable(name='gamma')
    L = f - gamma
    alpha_E_p = hierarchy_e_k([L] + list(gts) + list(eqs), k=p)
    ineq_dual_sigs = []
    summands = [L]
    for g in folded_gt:
        s_g_coeff = cl.Variable(name='s_' + str(g), shape=(alpha_E_p.shape[0],))
        s_g = Signomial(alpha_E_p, s_g_coeff)
        summands.append(-g * s_g)
        ineq_dual_sigs.append((s_g, g))
    eq_dual_sigs = []
    folded_eq = con_gen.up_to_q_fold_cons(eqs, q)
    for g in folded_eq:
        z_g_coeff = cl.Variable(name='z_' + str(g), shape=(alpha_E_p.shape[0],))
        z_g = Signomial(alpha_E_p, z_g_coeff)
        summands.append(-g * z_g)
        eq_dual_sigs.append((z_g, g))
    L = Signomial.sum(summands)
    return L, ineq_dual_sigs, eq_dual_sigs, gamma


def infer_domain(f, gts, eqs, check_feas=True):
    """
    Identify a subset of the constraints in ``gts`` and ``eqs`` which can be incorporated into
    conditional SAGE relaxations for signomials. Construct a SigDomain object from the inferred constraints.

    Parameters
    ----------
    f : Signomial
        The objective in a desired SAGE relaxation. This parameter is only used to determine
        the dimension of the set defined by constraints in ``gts`` and ``eqs``.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    check_feas : bool
        Indicates whether or not to verify that the returned SigDomain is nonempty.

    Returns
    -------
    X : SigDomain or None

    """
    conv_gt = con_gen.valid_posynomial_inequalities(gts)
    conv_eqs = con_gen.valid_monomial_equations(eqs)
    cl_cons = con_gen.clcons_from_standard_gprep(f.n, conv_gt, conv_eqs)
    if len(cl_cons) > 0:
        sigdom = SigDomain(f.n, coniclifts_cons=cl_cons, gts=conv_gt, eqs=conv_eqs, check_feas=check_feas)
        return sigdom
    else:
        return None


def hierarchy_e_k(sigs, k):
    alphas = [s.alpha for s in sigs]
    alpha = np.vstack(alphas)
    alpha = np.unique(alpha, axis=0)
    c = np.ones(shape=(alpha.shape[0],))
    s = Signomial(alpha, c)
    s = s ** k
    return s.alpha


def _check_kwargs(kwargs, allowed):
    for kw_key in kwargs:
        if kw_key not in allowed:
            raise ValueError('Keyword argument "' + kw_key + '" not recognized.')
