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
from sageopt.relaxations.sig_solution_recovery import local_refine,  is_feasible
from sageopt.relaxations.poly_solution_recovery import local_refine_polys_from_sigs

# import functions that don't have prefixes, and which this file needs to wrap.
from sageopt.relaxations.sage_sigs import sage_feasibility as _sig_sage_feasibility
from sageopt.relaxations.sage_polys import sage_feasibility as _poly_sage_feasibility
from sageopt.relaxations.sage_sigs import sage_multiplier_search as _sig_sage_mult_search
from sageopt.relaxations.sage_polys import sage_multiplier_search as _poly_sage_mult_search
from sageopt.relaxations.sage_sigs import conditional_sage_data as _sig_cond_sage_data
from sageopt.relaxations.sage_polys import conditional_sage_data as _poly_cond_sage_data


def sage_feasibility(f, X=None):
    if isinstance(f, Signomial):
        prob = _sig_sage_feasibility(f, X)
        return prob
    elif isinstance(f, Polynomial):
        prob = _poly_sage_feasibility(f, X)
        return prob
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')


def sage_multiplier_search(f, level=1, X=None):
    if isinstance(f, Polynomial):
        prob = _poly_sage_mult_search(f, level, X)
        return prob
    elif isinstance(f, Signomial):
        prob = _sig_sage_mult_search(f, level, X)
        return prob
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')


def conditional_sage_data(f, gts, eqs):
    if isinstance(f, Polynomial):
        X = _poly_cond_sage_data(f, gts, eqs)
        return X
    elif isinstance(f, Signomial):
        X = _sig_cond_sage_data(f, gts, eqs)
        return X
    else:
        raise ValueError('"f" must be a Signomial or Polynomial.')
