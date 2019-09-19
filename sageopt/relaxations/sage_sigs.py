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
from sageopt.symbolic.signomials import Signomial
from sageopt.relaxations import constraint_generators as con_gen
from sageopt.relaxations import symbolic_correspondences as sym_corr


def primal_sage_cone(sig, name, AbK, expcovers=None):
    if AbK is None:
        con = cl.PrimalSageCone(sig.c, sig.alpha, name, expcovers)
    else:
        A, b, K = AbK
        con = cl.PrimalCondSageCone(sig.c, sig.alpha, A, b, K, name, expcovers)
    return con


def relative_dual_sage_cone(primal_sig, dual_var, name, AbK, expcovers=None):
    if AbK is None:
        con = cl.DualSageCone(dual_var, primal_sig.alpha, name, primal_sig.c, expcovers)
    else:
        A, b, K = AbK
        con = cl.DualCondSageCone(dual_var, primal_sig.alpha, A, b, K, name, primal_sig.c, expcovers)
    return con


def sig_relaxation(f, form='dual', ell=0, X=None, mod_supp=None):
    """
    Construct a coniclifts Problem instance for producing a lower bound on

    .. math::

        f_X^{\\star} \doteq \min\{ f(x) \,:\, x \\in X \}

    where X = :math:`R^{\\texttt{f.n}}` by default.

    If ``form='dual'``, we can also attempt to recover solutions to the above problem.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    form : str
        Either ``form='primal'`` or ``form='dual'``.
    ell : int
        The level of the SAGE hierarchy. Must be nonnegative.
    X : dict
        If ``X`` is None, then we produce a bound on ``f`` over :math:`R^{\\texttt{f.n}}`.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.
    mod_supp : NumPy ndarray
        This parameter is only used when ``ell > 0``. If ``mod_supp`` is not None, then the rows of this
        array define the exponents of a positive definite modulating Signomial in the SAGE hierarchy.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts Problem which represents the SAGE relaxation, with given parameters.
        The relaxation can be solved by calling ``prob.solve()``.
    """
    if form.lower()[0] == 'd':
        prob = sig_dual(f, ell, X, mod_supp)
    elif form.lower()[0] == 'p':
        prob = sig_primal(f, ell, X, None, mod_supp)
    else:
        raise RuntimeError('Unrecognized form: ' + form + '.')
    return prob


def sig_dual(f, ell=0, X=None, modulator_support=None):
    """
    Construct a coniclifts Problem instance for producing a lower bound for ``f`` over the set defined by ``X``,
    and for attempting to recover optimal solutions to the problem ``min{ f(x) | x in X }``.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    ell : int
        The level of the SAGE hierarchy. Must be nonnegative.
    X : dict
        If ``X`` is None, then we produce a bound on ``f`` over :math:`R^{\\texttt{f.n}}`.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.
    modulator_support : NumPy ndarray
        If ``modulator_support`` is not None, then its rows define the exponents of a positive definite
        modulating Signomial in the SAGE hierarchy. This parameter is only used when ``ell > 0``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts Problem which represents the dual SAGE relaxation, with given parameters.
        The relaxation can be solved by calling ``prob.solve()``.

    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    f.remove_terms_with_zero_as_coefficient()
    # Signomial definitions (for the objective).
    if modulator_support is None:
        modulator_support = f.alpha
    t_mul = Signomial(modulator_support, np.ones(modulator_support.shape[0])) ** ell
    lagrangian = f - cl.Variable(name='gamma')
    metadata = {'f': f, 'lagrangian': lagrangian, 'modulator': t_mul, 'X': X}
    lagrangian = lagrangian * t_mul
    f_mod = f * t_mul
    # C_SAGE^STAR (v must belong to the set defined by these constraints).
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    con = relative_dual_sage_cone(lagrangian, v, name='Lagrangian SAGE dual constraint', AbK=X['AbK'])
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


def sig_primal(f, ell=0, X=None, additional_cons=None, modulator_support=None):
    """
    Construct a coniclifts Problem instance for producing a lower bound for ``f`` over the set defined by ``X``.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    ell : int
        The level of the SAGE hierarchy. Must be nonnegative.
    X : dict
        If ``X`` is None, then we produce a bound on ``f`` over :math:`R^{\\texttt{f.n}}`.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.
    modulator_support : NumPy ndarray
        If ``modulator_support`` is not None, then its rows define the exponents of a positive definite
        modulating Signomial in the SAGE hierarchy. This parameter is only used when ``ell > 0``.
    additional_cons: list of coniclifts Constraint objects
        Some primal SAGE polynomial relaxations can easily be transformed to primal SAGE signomial relaxations,
        by way of "signomial representatives". The signomial representatives are often accompanied by additional
        constraints. Those constraints may be passed as a list, via this argument. End-users are unlikely to use
        this argument.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts Problem which represents the primal SAGE relaxation, with given parameters.
        The relaxation can be solved by calling ``prob.solve()``.

    Notes
    -----
    Unlike ``sig_dual``, the ``sig_primal`` formulation can be stated in full generality without too much trouble.
    We define a multiplier signomial ``t`` (with the canonical choice ``t = Signomial(f.alpha, np.ones(f.m))``),
    then return problem data representing ::

        max  gamma
        s.t.    f_mod.c in C_{SAGE}(f_mod.alpha, X)
        where   f_mod := (t ** ell) * (f - gamma).

    Our implementation of Signomial objects allows Variables in the coefficient vector ``c``. As a result, the
    map from ``gamma`` to ``f_mod.c`` is an affine function that takes in a Variable and returns an Expression.
    This makes it very simple to represent ``f_mod.c in C_{SAGE}(f_mod.alpha, X)`` via coniclifts Constraints.
    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    f.remove_terms_with_zero_as_coefficient()
    if modulator_support is None:
        modulator_support = f.alpha
    t = Signomial(modulator_support, np.ones(modulator_support.shape[0]))
    gamma = cl.Variable(name='gamma')
    s_mod = (f - gamma) * (t ** ell)
    s_mod.remove_terms_with_zero_as_coefficient()
    con = primal_sage_cone(s_mod, name=str(s_mod), AbK=X['AbK'])
    constraints = [con]
    obj = gamma.as_expr()
    if additional_cons is not None:
        constraints += additional_cons
    prob = cl.Problem(cl.MAX, obj, constraints)
    cl.clear_variable_indices()
    return prob


def sage_feasibility(f, X=None, additional_cons=None):
    """
    Constructs a coniclifts maximization Problem which is feasible iff ``f`` admits an X-SAGE decomposition.

    Parameters
    ----------
    f : Signomial
        We want to test if this function admits an X-SAGE decomposition.
    X : dict
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.
    additional_cons : :obj:`list` of :obj:`sageopt.coniclifts.Constraint`
        This is mostly used for SAGE polynomials. When provided, it should be a list of Constraints over
        coniclifts Variables appearing in ``f.c``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
        A coniclifts maximization Problem. If ``f`` admits an X-SAGE decomposition, then we should have
        ``prob.value > -np.inf``, once ``prob.solve()`` has been called.
    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    f.remove_terms_with_zero_as_coefficient()
    con = primal_sage_cone(f, name=str(f), AbK=X['AbK'])
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
    X : dict
        If ``X`` is None, then we test nonnegativity of ``f`` over :math:`R^{\\texttt{f.n}}`.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    Notes
    -----
    This function provides an alternative to moving up the SAGE hierarchy, for the goal of certifying
    nonnegativity of a signomial ``f`` over some convex set ``X``.  In general, the approach is to introduce
    a signomial

        ``mult = Signomial(alpha_hat, c_tilde)``

    where the rows of ``alpha_hat`` are all ``level``-wise sums of rows from ``f.alpha``, and ``c_tilde``
    is a coniclifts Variable defining a nonzero X-SAGE function. Then we check if ``f_mod = f * mult``
    is X-SAGE for any choice of ``c_tilde``.
    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    f.remove_terms_with_zero_as_coefficient()
    constraints = []
    mult_alpha = hierarchy_e_k([f], k=level)
    c_tilde = cl.Variable(mult_alpha.shape[0], name='c_tilde')
    mult = Signomial(mult_alpha, c_tilde)
    constraints.append(cl.sum(c_tilde) >= 1)
    sig_under_test = mult * f
    con1 = primal_sage_cone(mult, name=str(mult), AbK=X['AbK'])
    con2 = primal_sage_cone(sig_under_test, name=str(sig_under_test), AbK=X['AbK'])
    constraints.append(con1)
    constraints.append(con2)
    prob = cl.Problem(cl.MAX, cl.Expression([0]), constraints)
    cl.clear_variable_indices()
    return prob


def sig_constrained_relaxation(f, gts, eqs, form='dual', p=0, q=1, ell=0, X=None):
    """
    Construct a coniclifts Problem instance representing a level-``(p, q, ell)`` SAGE relaxation
    for the signomial program

    .. math::

        \\begin{align*}
          \min\{ f(x) :~& g(x) \geq 0 \\text{ for } g \\in \\mathtt{gts}, \\\\
                       & g(x) = 0  \\text{ for } g \\in \\mathtt{eqs}, \\\\
                       & \\text{and } x \\in X \}
        \\end{align*}

    where X = :math:`R^{\\texttt{f.n}}` by default. When ``form='dual'``, a solution to this
    relaxation can be used  to help recover optimal solutions to the problem described above.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    form : str
        Either ``form='primal'`` or ``form='dual'``.
    p : int
        Controls the complexity of Lagrange multipliers in the primal formulation, and (equivalently) constraints in
        the dual formulation. The smallest value is ``p=0``, which corresponds to scalar Lagrange multipliers.
    q : int
        The number of folds applied to the constraints ``gts`` and ``eqs``. The smallest value is ``q=1``, which
        means "leave ``gts`` and ``eqs`` as-is."
    ell : int
        Controls the complexity of any modulator applied to the Lagrangian in the primal formulation, and
        (equivalently) constraints in the dual formulation. The smallest value is ``ell=0``, which means
        the primal Lagrangian must be a SAGE signomial.
    X : dict
        If ``X`` is None, then this parameter is ignored.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem
    """
    if form.lower()[0] == 'd':
        prob = sig_constrained_dual(f, gts, eqs, p, q, ell, X)
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

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    p : int
        Controls the complexity of Lagrange multipliers. Smallest value is ``p=0``, which corresponds to scalars.
    q : int
        The number of folds applied to the constraints ``gts`` and ``eqs``. Smallest value is ``q=1``, which means
        "leave ``gts`` and ``eqs`` as-is."
    ell : int
        Controls the complexity of any modulator applied to the Lagrangian.
        The smallest value is ``ell=0``, which means that the Lagrangian must be SAGE.
        ``ell=1`` means ``tilde_L = modulator * Lagrangian`` must be SAGE.
    X : dict
        If ``X`` is None, then this parameter is ignored.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, _, gamma = make_sig_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f] + list(gts) + list(eqs), k=1)
        modulator = Signomial(alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
    else:
        modulator = Signomial({(0,) * f.n: 1})
    metadata['modulator'] = modulator
    # The Lagrangian (after possible multiplication, as above) must be a SAGE signomial.
    con = primal_sage_cone(lagrangian, name='Lagrangian is SAGE', AbK=X['AbK'])
    constrs = [con]
    #  Lagrange multipliers (for inequality constraints) must be SAGE signomials.
    expcovers = None
    for i, (s_h, _) in enumerate(ineq_lag_mults):
        con_name = 'SAGE multiplier for signomial inequality # ' + str(i)
        con = primal_sage_cone(s_h, name=con_name, AbK=X['AbK'], expcovers=expcovers)
        expcovers = con.ech.expcovers  # only * really * needed in first iteration, but keeps code flat.
        constrs.append(con)
    # Construct the coniclifts Problem.
    prob = cl.Problem(cl.MAX, gamma, constrs)
    prob.metadata = metadata
    cl.clear_variable_indices()
    return prob


def sig_constrained_dual(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the SAGE-(p, q, ell) dual problem for the signomial program

        min{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where X = :math:`R^{\\texttt{f.n}}` by default.

    Parameters
    ----------
    f : Signomial
        The objective function to be minimized.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomials
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.
    p : int
        Controls the complexity of Lagrange multipliers in the primal problem, and in an turn the complexity of
        constraints in this dual problem. The smallest value is ``p=0``, which corresponds to scalar Lagrange
        multipliers in the primal problem, and linear inequality constraints in this dual problem.
    q : int
        The number of folds applied to the constraints ``gts`` and ``eqs``. Smallest value is ``q=1``, which means
        "leave ``gts`` and ``eqs`` as-is."
    ell : int
        Controls the complexity of any modulator applied to the Lagrangian in the primal problem.
        The smallest value is ``ell=0``, which means the primal Lagrangian is unchanged / not modulated.
    X : dict
        If ``X`` is None, then this parameter is ignored.
        If ``X`` is a dict, then it must contain three fields: ``'AbK'``, ``'gts'``, and ``'eqs'``. For almost all
        applications, the appropriate dict ``X`` can be generated for you by calling ``conditional_sage_data(...)``.

    Returns
    -------
    prob : sageopt.coniclifts.Problem

    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, eq_lag_mults, _ = make_sig_lagrangian(f, gts, eqs, p=p, q=q)
    metadata = {'lagrangian': lagrangian, 'f': f, 'gts': gts, 'eqs': eqs, 'level': (p, q, ell), 'X': X}
    if ell > 0:
        alpha_E_1 = hierarchy_e_k([f] + list(gts) + list(eqs), k=1)
        modulator = Signomial(alpha_E_1, np.ones(alpha_E_1.shape[0])) ** ell
        lagrangian = lagrangian * modulator
        f = f * modulator
    else:
        modulator = Signomial({(0,) * f.n: 1})
    metadata['modulator'] = modulator
    # In primal form, the Lagrangian is constrained to be a SAGE signomial.
    # Introduce a dual variable "v" for this constraint.
    v = cl.Variable(shape=(lagrangian.m, 1), name='v')
    con = relative_dual_sage_cone(lagrangian, v, name='Lagrangian SAGE dual constraint', AbK=X['AbK'])
    constraints = [con]
    expcovers = None
    for i, (s_h, h) in enumerate(ineq_lag_mults):
        # These generalized Lagrange multipliers "s_h" are SAGE signomials.
        # For each such multiplier, introduce an appropriate dual variable "v_h", along
        # with constraints over that dual variable.
        v_h = cl.Variable(name='v_' + str(h), shape=(s_h.m, 1))
        con_name = 'SAGE dual for signomial inequality # ' + str(i)
        con = relative_dual_sage_cone(s_h, v_h, name=con_name, AbK=X['AbK'], expcovers=expcovers)
        expcovers = con.ech.expcovers  # only * really * needed in first iteration, but keeps code flat.
        constraints.append(con)
        h = h * modulator
        c_h = sym_corr.moment_reduction_array(s_h, h, lagrangian)
        constraints.append(c_h @ v == v_h)
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
    alpha_E_p = hierarchy_e_k([f] + list(gts) + list(eqs), k=p)
    ineq_dual_sigs = []
    for g in folded_gt:
        s_g_coeff = cl.Variable(name='s_' + str(g), shape=(alpha_E_p.shape[0],))
        s_g = Signomial(alpha_E_p, s_g_coeff)
        L -= s_g * g
        ineq_dual_sigs.append((s_g, g))
    eq_dual_sigs = []
    folded_eq = con_gen.up_to_q_fold_cons(eqs, q)
    for g in folded_eq:
        z_g_coeff = cl.Variable(name='z_' + str(g), shape=(alpha_E_p.shape[0],))
        z_g = Signomial(alpha_E_p, z_g_coeff)
        L -= z_g * g
        eq_dual_sigs.append((z_g, g))
    return L, ineq_dual_sigs, eq_dual_sigs, gamma


def conditional_sage_data(f, gts, eqs, check_feas=True):
    """
    Identify a subset of the constraints in ``gts`` and ``eqs`` which can be incorporated into
    conditional SAGE relaxations. Generate conic data that relaxation-constructors will need
    in downstream applications.

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
        Indicates whether or not to verify that the returned conic system is feasible.

    Returns
    -------
    X : dict

        ``X`` is keyed by three strings: ``'AbK'``, ``'gts'``, and ``'eqs'``. Refer to the Notes
        for discussion on the values associated with these keys.

    Notes
    -----
    ``X['gts']`` is a list of Signomials. Every ``g in X['gts']`` has an efficient
    convex representation of ``{x : g(x) >= 0}``, and the intersection of all these
    sets contains ``{x : g(x) >= 0 for all g in gts}``. There is one signomial ``g in X['gts']``
    for every ``g in gts`` which has one positive coefficient.

    ``X['eqs]`` is a list of Signomials. Every ``g in X['eqs']`` has an efficient
    convex representation of ``{x : g(x) == 0}``, and the intersection of all these sets
    contains ``{x : g(x) == 0 for all g in eqs}``. There is one signomial ``g in X['eqs']``
    for every ``g in eqs`` which has exactly two coefficients (one positive, one negative).

    ``X['AbK']`` is a coniclifts-standard representation of the feasible set cut out by
    ``X['gts']`` and ``X['eqs']``. We can check membership in ``X['AbK']`` by evaluating
    the functions in ``X['gts']`` and ``X['eqs']``, and checking that the results are
    nonnegative and zero respectively.
    """
    x = cl.Variable(shape=(f.n,), name='x')
    coniclift_cons = []
    conv_gt = con_gen.valid_posynomial_inequalities(gts)
    for g in conv_gt:
        nonconst_selector = np.ones(shape=(g.m,), dtype=bool)
        nonconst_selector[g.constant_location()] = False
        if g.m > 2:
            cst = g.c[~nonconst_selector]
            alpha = g.alpha[nonconst_selector, :]
            c = -g.c[nonconst_selector]
            expr = cl.weighted_sum_exp(c, alpha @ x)
            coniclift_cons.append(expr <= cst)
        elif g.m == 2:
            expr = g.alpha[nonconst_selector, :] @ x
            cst = np.log(g.c[~nonconst_selector] / abs(g.c[nonconst_selector]))
            coniclift_cons.append(expr <= cst)
    conv_eqs = con_gen.valid_monomial_equations(eqs)
    for g in conv_eqs:
        # g is of the form c1 - c2 * exp(a.T @ x) == 0, where c1, c2 > 0
        cst_loc = g.constant_location()
        non_cst_loc = 1 - cst_loc
        rhs = np.log(g.c[cst_loc] / abs(g.c[non_cst_loc]))
        coniclift_cons.append(g.alpha[non_cst_loc, :] @ x == rhs)
    if len(coniclift_cons) > 0:
        A, b, K, _, _, _ = cl.compile_constrained_system(coniclift_cons)
        if check_feas:
            x = cl.Variable(shape=(A.shape[1],), name='temp_x')
            A_dense = A.toarray()
            cons = [cl.PrimalProductCone(A_dense @ x + b, K)]
            prob = cl.Problem(cl.MIN, cl.Expression([0]), cons)
            prob.solve(verbose=False, solver='ECOS')
            if not prob.value < 1e-7:
                msg1 = 'Inferred constraints could not be verified as feasible.\n'
                msg2 = 'Feasibility problem\'s status: ' + prob.status + '\n'
                msg3 = 'Feasibility problem\'s  value: ' + str(prob.value) + '\n'
                msg4 = 'The objective was "minimize 0"; we expect problem value < 1e-7. \n'
                msg = msg1 + msg2 + msg3 + msg4
                raise RuntimeError(msg)
        AbK = (A, b, K)
        X = {'AbK': AbK, 'gts': conv_gt, 'eqs': conv_eqs}
        return X
    else:
        X = {'AbK': None, 'gts': [], 'eqs': []}
        return X


def hierarchy_e_k(sigs, k):
    alpha_tups = sum([list(s.alpha_c.keys()) for s in sigs], [])
    alpha_tups = set(alpha_tups)
    s = Signomial(dict([(a, 1.0) for a in alpha_tups]))
    s = s ** k
    return s.alpha
