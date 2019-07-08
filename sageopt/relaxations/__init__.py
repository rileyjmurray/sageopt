from sageopt.relaxations import sage_polys
from sageopt.relaxations import sage_sigs
from sageopt.relaxations import constraint_generators

from sageopt.symbolic.signomials import Signomial
from sageopt.symbolic.polynomials import Polynomial

# import function names with the sig_ prefix.
from sageopt.relaxations.sage_sigs import sig_primal, sig_dual, sig_constrained_primal, sig_constrained_dual
from sageopt.relaxations.sig_solution_recovery import sig_solrec

# import function names with the poly_ prefix.
from sageopt.relaxations.sage_polys import poly_primal, poly_dual, poly_constrained_primal, poly_constrained_dual
from sageopt.relaxations.poly_solution_recovery import poly_solrec

# import remaining functions for solution recovery
from sageopt.relaxations.sig_solution_recovery import local_refine, is_feasible
from sageopt.relaxations.poly_solution_recovery import local_refine_polys_from_sigs

# import functions that don't have prefixes, and which this file needs to wrap.
from sageopt.relaxations.sage_sigs import sage_feasibility as _sig_sage_feasibility
from sageopt.relaxations.sage_polys import sage_feasibility as _poly_sage_feasibility
from sageopt.relaxations.sage_sigs import sage_multiplier_search as _sig_sage_mult_search
from sageopt.relaxations.sage_polys import sage_multiplier_search as _poly_sage_mult_search
from sageopt.relaxations.sage_sigs import conditional_sage_data as _sig_cond_sage_data
from sageopt.relaxations.sage_polys import conditional_sage_data as _poly_cond_sage_data


def sage_feasibility(f, X=None):
    """
    Return a coniclifts Problem ``prob`` with ``prob.value > -np.inf`` iff
    ``f`` admits an  X-SAGE decomposition.

    Parameters
    ----------
    f : Signomial or Polynomial
        We want to know if this function admits an X-SAGE decomposition
    X : dict or None
        If ``X`` is a dict, then it must be generated in accordance with the
        function ``conditional_sage_data``.

    Returns
    -------
    A coniclifts Problem object, with objective "maximize 0", and constraints
    requiring that ``f`` is X-SAGE. If the argument ``X`` is ``None``, then
    we test if ``f`` is R^n-SAGE.

    Notes
    -----
    This function is simply a wrapper around two functions of the same name,
    which were written for cases where ``f`` is a Signomial or Polynomial.

    The signomial and polynomial cases are implemented in
    ``sageopt.relaxations.sage_sigs.sage_feasibility`` and
    ``sageopt.relaxations.sage_polys.sage_feasibility`` respectively.

    """
    if isinstance(f, Signomial):
        prob = _sig_sage_feasibility(f, X)
        return prob
    elif isinstance(f, Polynomial):
        prob = _poly_sage_feasibility(f, X)
        return prob
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')


def sage_multiplier_search(f, level=1, X=None):
    """
    Return a coniclifts Problem ``prob``, where ``prob.value > -np.inf`` iff ``f``
    can be certified as nonnegative over ``X`` (using a modulator of complexity
    determined by ``level``).

    Parameters
    ----------
    f : Signomial or Polynomial
        We are looking for a certificate that ``f`` is nonnegative over ``X``.
    level : int
        A positive integer. This determines the complexity of a modulator ``g``
        which is positive over ``X``.
    X : dict or None
        If ``X`` is a dict, then it must be generated in accordance with the
        function ``conditional_sage_data``. If ``X`` is None, then we certify
        nonnegativity over R^n.

    Returns
    -------
    prob : coniclifts.Problem
        A problem with objective "maximize 0", and constraints that both ``g``
        and ``f * g`` are X-SAGE. The main variables in this problem are the coefficients
        of ``g``. The complexity of ``g`` is determined by ``level``.

    Notes
    -----
    This function is simply a wrapper around two functions of the same name,
    which were written for cases where ``f`` is a Signomial or Polynomial.

    The signomial and polynomial cases are implemented in
    ``sageopt.relaxations.sage_sigs.sage_multiplier_search`` and
    ``sageopt.relaxations.sage_polys.sage_multiplier_search`` respectively.

    """
    if isinstance(f, Polynomial):
        prob = _poly_sage_mult_search(f, level, X)
        return prob
    elif isinstance(f, Signomial):
        prob = _sig_sage_mult_search(f, level, X)
        return prob
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')


def conditional_sage_data(f, gts, eqs):
    """
    Infer (and construct a representation for) a tractable set ``X`` which
    is contained within { x : g(x) >= 0 for g in gts } and
    {x : g(x) == 0 for g in eqs}. For use in conditional SAGE relaxations.

    Parameters
    ----------
    f : Signomial or Polynomial
        This is only used to determine the dimension of the output ``X``
    gts : list of Signomials, or list of Polynomials
        Desired inequality constraint functions.
    eqs : list of Signomials, or list of Polynomials
        Desired equality constraint functions.

    Returns
    -------
    X : dict
        ``X`` has three key-value pairs.

        ``X['gts']`` is a list of functions which define inequality constraints,
        and ``X['eqs']`` is a list of functions which define equality constraints.

        The third key depends on if provided functions were Signomials or Polynomials.
        If the given functions were Signomials, then ``X`` has a key  ``X['AbK']``.
        If the given functions were Polynomials, then ``X`` has a key ``X['log_AbK']``.

    Notes
    -----
    This is a wrapper around two functions of the same name,
    which generate ``X`` in the signomial and polynomial cases.

    The signomial and polynomial cases are implemented in
    ``sageopt.relaxations.sage_sigs.sage_feasibility`` and
    ``sageopt.relaxations.sage_polys.sage_feasibility`` respectively.

    Refer to those functions for detailed documentation.

    """
    if isinstance(f, Polynomial):
        X = _poly_cond_sage_data(f, gts, eqs)
        return X
    elif isinstance(f, Signomial):
        X = _sig_cond_sage_data(f, gts, eqs)
        return X
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')
