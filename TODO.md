This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Improve management of sage cone compilation settings

To view the setting, a user has to trace back far into coniclifts
package structure, and access a protected variable. That's dumb.
Also, the place where the user accesses the setting is completely
removed from where the user *sets* the setting. Also dumb.

PrimalSageCone and DualSageCone objects should cache the compilation
settings they observe during construction. This can mostly be done
within the ExpCoverHelper class. However the PrimalSageCone class
also needs to track if it uses age-sum-leq-c or age-sum-eq-c.

## Consider moving infer_domain to symbolic

Right now, SigDomain and PolyDomain objects are part of the symbolic
subpackage, but infer_domain is part of the relaxations subpackage.
Conceptually, it seems as though these should be in a common place.

## Add keyword argument checks in sage_sigs.py and sage_polys.py.

Make sure to raise a value error if an unrecognized keyword argument
is supplied. This is necessary to catch typos in keyword arguments
which aren't part of the function signature.

## Web documentation

1. Suggest running nosetests, after installing from pip
2. Add documentation for keyword arguments to Problem.solve()

## Expose MOSEK solver options to prob.solve()

Need to expose MOSEK solver settings. Ones of practical interest
are ``mosek.dparam.intpnt_co_tol_near_rel`` and
``mosek.iparam.intpnt_scaling``.

Send sage_benchmarks primal problem 2, params (p=0, q=3, ell=0, nontriv
X) to MOSEK. With the default scaling, it returns a significantly
infeasible solution.

## sageopt.interop.gpkit

Write tests. Setup a continuous-integration environment which
installs GPKit. Add web documentation.


## Signomial and Polynomial class refactoring

Make changes on web-documentation, and "Examples".

## Fix a bug for signomial construction

if ``f`` is a Signomial, and ``arr`` is a numpy array,
then ``temp = -arr + f`` creates a numpy array of the
expected Signomial objects, but ``temp = f - arr`` raises a
``ValueError`` when ``upcast_to_signomial`` fails.
