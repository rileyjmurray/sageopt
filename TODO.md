This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Consider moving infer_domain to symbolic

Right now, SigDomain and PolyDomain objects are part of the symbolic
subpackage, but infer_domain is part of the relaxations subpackage.
Conceptually, it seems as though these should be in a common place.

## Web documentation

1. Add examples for CVXPY and GPKit integration.

## Send a bad problem case to MOSEK devs.

sage_benchmarks primal problem 2, params (p=0, q=3, ell=0, nontriv
X). With the default scaling, it returns a significantly
infeasible solution.

## sageopt.interop.cvxpy
Write tests. Setup a continuous-integration environment which
installs CVXPY.

## ECOS interface

Need to parse input dict for more solver options (like accuracy tolerances). 

## coniclifts

Create an attribute like ``arg_sig`` for PrimalSageCone objects, which
returns a Signomial object matching the one specified by arguments to 
the PrimalSageCone constraint. Also, create an instance method which
looks at the numeric value of the coefficient vector subject to a PrimalSageCone
constraint (e.g. after an underlying optimization problem has been solved),
and then solves the SAGE feasibility problem again for that vector, this time
using information on the fixed sign pattern. This will result in fewer summand
AGE functions. Could add an argument to this function which encourages
sparsity in the solution by using mixed-integer auxiliary variables. 


## sage_polys.py

Polynomial SAGE constraints stand to benefit significantly from providing
the ExpCovers argument to PrimalSageCone and DualSageCone objects, since
we know the effective signs of coefficients corresponding to non-even
exponents (even when those coefficients are symbolic coniclifts Expressions).
This can be addressed in ``sage_polys.py:primal_sage_poly_cone``. No need
to modify the signomial representative. Just compute some basic expcovers there,
and pass to the constructor. (Okay actually I would probably also want to change
how ExpCoverHelper objects so that they compute "default" expcovers even expcovers
are provided by the user; then the ExpCoverHelper can possibly simplify the user
-provided expcovers.)
