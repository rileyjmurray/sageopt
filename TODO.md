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
