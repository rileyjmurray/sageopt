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


def primal_sage_cone(sig, name, AbK):
    if AbK is None:
        con = cl.PrimalSageCone(sig.c, sig.alpha, name)
    else:
        A, b, K = AbK
        con = cl.PrimalCondSageCone(sig.c, sig.alpha, A, b, K, name)
    return con


def relative_dual_sage_cone(primal_sig, dual_var, name, AbK):
    if AbK is None:
        con = cl.DualSageCone(dual_var, primal_sig.alpha, name, primal_sig.c)
    else:
        A, b, K = AbK
        con = cl.DualCondSageCone(dual_var, primal_sig.alpha, A, b, K, name, primal_sig.c)
    return con


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
        If ``X`` is None, then we produce a bound on ``f`` over R^n.
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
    prob.associated_data = metadata
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
        If ``X`` is None, then we produce a bound on ``f`` over R^n.
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
    :param f: a Signomial object
    :param X: None, or a dictionary with three keys 'AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".
    :param additional_cons: a list of coniclifts Constraint objects over the variables in f.c
    (unless you are working with SAGE polynomials, there likely won't be any of these).

    :return: coniclifts maximization Problem which is feasible iff f.c in C_{SAGE}(f.alpha, X),
    where X = R^(f.n) by default.
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
    :param f: a Signomial object
    :param level: a nonnegative integer; controls the complexity of the search space for SAGE multipliers.
    :param X: None, or a dictionary with three keys 'AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".

    :return: a coniclifts Problem that is feasible iff f * mult is X-SAGE for some X-SAGE multiplier
    signomial "mult". X = R^(f.n) by default.

    This function provides an alternative to moving up the SAGE hierarchy, for the goal of certifying
    nonnegativity of a signomial "f" over some convex set "X."  In general, the approach is to introduce
    a signomial

        mult = Signomial(alpha_hat, c_tilde)

    where the rows of alpha_hat are all level-wise sums of rows from f.alpha, and c_tilde is a coniclifts
    Variable defining a nonzero X-SAGE function. Then we check if f_mod := f * mult is X-SAGE for any
    choice of c_tilde.
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


def sig_constrained_primal(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the SAGE-(p, q, ell) primal problem for the signomial program

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where X = R^(f.n) by default.

    :param f: a Signomial.
    :param gts: a list of Signomials.
    :param eqs: a list of Signomials.
    :param p: a nonnegative integer.
        Controls the complexity of Lagrange multipliers. p=0 corresponds to scalars.
    :param q: a positive integer.
        The number of folds applied to the constraints "gts" and "eqs". p=1 means "leave gts and eqs as-is."
    :param ell: a nonnegative integer.
        Controls the complexity of any modulator applied to the Lagrangian. ell=0 means that
        the Lagrangian must be SAGE. ell=1 means "tilde_L := modulator * Lagrangian" must be SAGE.
    :param X: None, or a dictionary with three keys 'AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".

    :return: The primal form SAGE-(p, q, ell) relaxation for the given signomial program.
    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, _, gamma = make_lagrangian(f, gts, eqs, p=p, q=q)
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
    for i, (s_h, _) in enumerate(ineq_lag_mults):
        con_name = 'SAGE multiplier for signomial inequality # ' + str(i)
        con = primal_sage_cone(s_h, name=con_name, AbK=X['AbK'])
        constrs.append(con)
    # Construct the coniclifts Problem.
    prob = cl.Problem(cl.MAX, gamma, constrs)
    prob.associated_data = metadata
    cl.clear_variable_indices()
    return prob


def sig_constrained_dual(f, gts, eqs, p=0, q=1, ell=0, X=None):
    """
    Construct the SAGE-(p, q, ell) dual problem for the signomial program

        inf{ f(x) : g(x) >= 0 for g in gts,
                    g(x) == 0 for g in eqs,
                    and x in X }

    where X = R^(f.n) by default.

    :param f: a Signomial.
    :param gts: a list of Signomials.
    :param eqs: a list of Signomials.
    :param p: a nonnegative integer.
        Controls the complexity of Lagrange multipliers in the primal problem (p=0 corresponds to scalars),
        and in turn the complexity of constraints in this dual problem (p=0 corresponds to linear inequalities).
    :param q: a positive integer.
        The number of folds applied to the constraints "gts" and "eqs". p=1 means "leave gts and eqs as-is."
    :param ell: a nonnegative integer.
        Controls the complexity of any modulator applied to the Lagrangian in the primal problem.
        ell=0 means that the Lagrangian is unchanged / not modulated.
    :param X: None, or a dictionary with three keys 'AbK', 'gts', 'eqs' such as that generated by
     the function "conditional_sage_data(...)".

    :return: The dual form SAGE-(p, q, ell) relaxation for the given signomial program.
    """
    if X is None:
        X = {'AbK': None, 'gts': [], 'eqs': []}
    lagrangian, ineq_lag_mults, eq_lag_mults, _ = make_lagrangian(f, gts, eqs, p=p, q=q)
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
    for i, (s_h, h) in enumerate(ineq_lag_mults):
        # These generalized Lagrange multipliers "s_h" are SAGE signomials.
        # For each such multiplier, introduce an appropriate dual variable "v_h", along
        # with constraints over that dual variable.
        v_h = cl.Variable(name='v_' + str(h), shape=(s_h.m, 1))
        con_name = 'SAGE dual for signomial inequality # ' + str(i)
        con = relative_dual_sage_cone(s_h, v_h, name=con_name, AbK=X['AbK'])
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
    prob.associated_data = metadata
    cl.clear_variable_indices()
    return prob


def make_lagrangian(f, gts, eqs, p, q):
    """
    Given a problem

        inf_{x in X}{ f(x) : g(x) >= 0 for g in gts, g(x) == 0 for g in eqs },

    construct the q-fold constraints "folded_gts" and "folded_eqs," and the Lagrangian

        L = f - gamma - sum_{g in folded_gts} s_g * g - sum_{g in folded_eqs} z_g * g

    where gamma and the coefficients on Signomials s_g / z_g are coniclifts Variables.

    The values returned by this function are used to construct constrained SAGE relaxations.
    The basic primal SAGE relaxation is obtained by maximizing gamma, subject to the constraint
    that L and each s_g are SAGE functions. The dual SAGE relaxation is obtained by symbolically
    applying conic duality to the primal.

    :param f: a Signomial.
    :param gts: a list of Signomials.
    :param eqs: a list of Signomials.
    :param p: a nonnegative integer. Controls the complexity of s_g and z_g.
    :param q: a positive integer. The number of folds of constraints "gts" and "eqs".

    :return: L, ineq_dual_sigs, eq_dual_sigs, gamma.

        L : a Signomial object, whose coefficient vector is an an affine expression of coniclifts Variables.

        ineq_dual_sigs : a list of pairs of Signomial objects. If the pair (s1, s2) is in this list, then
            s1 is a generalized Lagrange multiplier to the constraint that s2(x) >= 0.

        eq_dual_sigs : a list of pairs of Signomial objects. If the pair (s1, s2) is in this list, then
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


def conditional_sage_data(f, gts, eqs):
    """

    Parameters
    ----------
    f : Signomial
        The objective in a desired optimization problem. This parameter is only used to determine
        the dimension of the set defined by constraints in ``gts`` and ``eqs``.
    gts : list of Signomials
        For every ``g in gts``, there is a desired constraint that variables ``x`` satisfy ``g(x) >= 0``.
    eqs : list of Signomial
        For every ``g in eqs``, there is a desired constraint that variables ``x`` satisfy ``g(x) == 0``.

    Returns
    -------
    X : dict

        ``X`` is keyed by three strings: ``'AbK'``, ``'gts'``, and ``'eqs'``.

        ``X['gts']`` is a list of Signomials so that every ``g in X['gts']`` has an efficient
        convex representation of {x : g(x) >= 0}. The intersection of all of these sets
        is contained within {x : g(x) >= 0 for all g in gts}. ``X['eqs']`` is defined
        similarly, but for equality constraints instead of inequality constraints.

        If there are no constraints defined by ``X['gts']`` and ``X['eqs']``, then ``X['AbK']`` is None.

        Otherwise, ``X['AbK']`` is a conic representation of the feasible set cut out by
        ``X['gts']`` and ``X['eqs']``. The conic representation is a triple ``X['AbK'] = (A, b, K)``,
        where ``A`` is a SciPy sparse matrix, ``b`` is a numpy 1d array, and ``K`` is a list of
        coniclifts Cone objects. The number of columns for ``A`` in ``X['AbK']`` will always
        be at least ``f.n``. If the number of columns is greater than ``f.n``, then the first ``f.n``
        columns of ``A`` correspond (in order!) to the variables over which ``f`` is defined. Any
        remaining columns are auxiliary variables needed to represent ``X`` in coniclifts primitives.

    Notes
    -----
        This function essentially defines the requirements for ``X`` which may be passed to
        conditional SAGE relaxations defined in this python module.

        If a user wants to define their own ``X`` without calling this function, that is doable.
        A possible use-case here is when ``X`` has a simple convex description, but not a simple
        description in terms of signomial inequality and equality constraints.

        Suppose for example that we want ``X`` to represent the ell-2 unit ball in R^{f.n}.
        This can be accomplished as follows ::

            import sageopt.coniclifts as cl
            import numpy as np
            x = cl.Variable(shape=(f.n,), name='x')
            constraints = [1 >= cl.vector2norm(x)]
            A, b, K, _, _ = cl.compile_constrained_system(constraints)
            cl.clear_variable_indices()
            my_gts = [lambda dummy_x: 1 - np.linalg.norm(dummy_x, ord=2)]
            my_eqs = []
            X = {'AbK': (A, b, K), 'gts': my_gts, 'eqs': my_eqs}

        The message of this example is that ``X['gts']`` and ``X['eqs']`` don't actually need
        to be lists of Signomials. They just need to be callable functions which define membership
        in ``X['AbK]``. To really hit this home, functions in ``X['gts']`` don't even need to be
        continuous! For example, if you wanted to test membership in the ell-2 ball without
        allowing even the smallest constraint violation, you could use the function ::

            my_gts = [lambda dummy_x: 1 if 1 >= np.linalg.norm(dummy_x, ord=2) else -np.inf]

        and every important feature of sageopt's signomial relaxations would work as expected.
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
        else:
            raise RuntimeError('Trivial (or infeasible) signomial constraint.')
    conv_eqs = con_gen.valid_monomial_equations(eqs)
    for g in conv_eqs:
        # g is of the form c1 - c2 * exp(a.T @ x) == 0, where c1, c2 > 0
        cst_loc = g.constant_location()
        non_cst_loc = 1 if cst_loc == 0 else 1
        rhs = np.log(g.c[cst_loc] / abs(g.c[non_cst_loc]))
        coniclift_cons.append(g.alpha[non_cst_loc, :] @ x == rhs)
    if len(coniclift_cons) > 0:
        A, b, K, _, _ = cl.compile_constrained_system(coniclift_cons)
        cl.clear_variable_indices()
        AbK = (A, b, K)
        X = {'AbK': AbK, 'gts': conv_gt, 'eqs': conv_eqs}
        return X
    else:
        cl.clear_variable_indices()
        X = {'AbK': None, 'gts': [], 'eqs': []}
        return X


def hierarchy_e_k(sigs, k):
    alpha_tups = sum([list(s.alpha_c.keys()) for s in sigs], [])
    alpha_tups = set(alpha_tups)
    s = Signomial(dict([(a, 1.0) for a in alpha_tups]))
    s = s ** k
    return s.alpha
