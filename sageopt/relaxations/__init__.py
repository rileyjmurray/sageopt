from sageopt.relaxations import sage_polys
from sageopt.relaxations import sage_sigs
from sageopt.relaxations import constraint_generators

from sageopt.symbolic.signomials import Signomial
from sageopt.symbolic.polynomials import Polynomial

# import function names with the sig_ prefix.
from sageopt.relaxations.sage_sigs import sig_primal, sig_dual, sig_constrained_primal, sig_constrained_dual
from sageopt.relaxations.sage_sigs import sig_relaxation, sig_constrained_relaxation
from sageopt.relaxations.sig_solution_recovery import sig_solrec

# import function names with the poly_ prefix.
from sageopt.relaxations.sage_polys import poly_primal, poly_dual, poly_constrained_primal, poly_constrained_dual
from sageopt.relaxations.sage_polys import poly_relaxation, poly_constrained_relaxation
from sageopt.relaxations.poly_solution_recovery import poly_solrec

# import remaining functions for solution recovery
from sageopt.relaxations.sig_solution_recovery import local_refine, is_feasible
from sageopt.relaxations.poly_solution_recovery import local_refine_polys_from_sigs

# import functions that don't have prefixes, and which this file needs to wrap.
from sageopt.relaxations.sage_sigs import sage_feasibility as _sig_sage_feasibility
from sageopt.relaxations.sage_polys import sage_feasibility as _poly_sage_feasibility
from sageopt.relaxations.sage_sigs import sage_multiplier_search as _sig_sage_mult_search
from sageopt.relaxations.sage_polys import sage_multiplier_search as _poly_sage_mult_search
from sageopt.relaxations.sage_sigs import infer_domain as _sig_cond_sage_data
from sageopt.relaxations.sage_polys import infer_domain as _poly_cond_sage_data


def sage_feasibility(f, X=None):
    """
    Construct a Problem for checking if ``f`` admits an  X-SAGE decomposition.

    Parameters
    ----------
    f : Signomial or Polynomial
        We want to know if this function admits an X-SAGE decomposition
    X : SigDomain or PolyDomain
        Default to :math:`X = \\mathbb{R}^{\\texttt{f.n}}`.

    Returns
    -------
    prob : coniclifts.Problem
        Has objective "maximize 0", and constraints that ``f`` is X-SAGE.
    """
    if isinstance(f, Polynomial):
        prob = _poly_sage_feasibility(f, X)
        return prob
    elif isinstance(f, Signomial):
        prob = _sig_sage_feasibility(f, X)
        return prob
    else:  # pragma: no cover
        raise ValueError('"f" must be a Signomial or Polynomial.')


def sage_multiplier_search(f, level=1, X=None):
    """
    Construct a Problem for an attempt to certify that ``f`` is X-nonnegative.

    Parameters
    ----------
    f : Signomial or Polynomial
        We are looking for a certificate that ``f`` is nonnegative over ``X``.
    level : int
        A positive integer. This determines the complexity of a modulator ``g``
        which is positive over ``X``.
    X : SigDomain or PolyDomain
        Default to :math:`X = \\mathbb{R}^{\\texttt{f.n}}`.

    Returns
    -------
    prob : coniclifts.Problem
        A problem with objective "maximize 0", and constraints that both ``g``
        and ``f * g`` are X-SAGE. The main variables in this problem are the coefficients
        of ``g``. The complexity of ``g`` is determined by ``level``.
    """
    if isinstance(f, Polynomial):
        prob = _poly_sage_mult_search(f, level, X)
        return prob
    elif isinstance(f, Signomial):
        prob = _sig_sage_mult_search(f, level, X)
        return prob
    else:  # pragma: no cover
        raise ValueError('"f" must be a Signomial or Polynomial.')


def infer_domain(f, gts, eqs, check_feas=True):
    """
    Infer (and construct a representation for) a tractable set ``X`` which
    is contains { x : g(x) >= 0 for g in gts } and
    {x : g(x) == 0 for g in eqs}. For use in conditional SAGE relaxations.

    Parameters
    ----------
    f : Signomial or Polynomial
        This is only used to determine the dimension of the output ``X``
    gts : list of Signomials, or list of Polynomials
        Desired inequality constraint functions.
    eqs : list of Signomials, or list of Polynomials
        Desired equality constraint functions.
    check_feas : bool
        Indicates whether or not to verify that the returned conic system is feasible.

    Returns
    -------
    X : SigDomain or PolyDomain
    """
    if isinstance(f, Polynomial):
        X = _poly_cond_sage_data(f, gts, eqs, check_feas)
        return X
    elif isinstance(f, Signomial):
        X = _sig_cond_sage_data(f, gts, eqs, check_feas)
        return X
    else:  # pragma: no cover
        raise ValueError('"f" must be a Signomial or Polynomial.')
