This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Consider moving infer_domain to symbolic

Right now, SigDomain and PolyDomain objects are part of the symbolic
subpackage, but infer_domain is part of the relaxations subpackage.
Conceptually, it seems as though these should be in a common place.

## Web documentation

1. Suggest running nosetests, after installing from pip
2. Add examples for CVXPY and GPKit integration.

## Send a bad problem case to MOSEK devs.

sage_benchmarks primal problem 2, params (p=0, q=3, ell=0, nontriv
X). With the default scaling, it returns a significantly
infeasible solution.

## sageopt.interop.cvxpy
Write tests. Setup a continuous-integration environment which
installs CVXPY.

## Fix a bug for signomial construction

if ``f`` is a Signomial, and ``arr`` is a numpy array,
then ``temp = -arr + f`` creates a numpy array of the
expected Signomial objects, but ``temp = f - arr`` raises a
``ValueError`` when ``upcast_to_signomial`` fails.
