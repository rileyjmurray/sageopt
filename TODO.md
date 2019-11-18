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

Make sure to raise a runtime error if an unrecognized keyword argument
is supplied. This is necessary to catch typos in keyword arguments
which aren't part of the function signature.

## Add documentation for keyword arguments to Problem.solve()

Right now there isn't any. Cover generic keyword arguments (e.g.
``verbose``, ``cache_apply_data``), and solver-specific keyword
arguments (e.g. ``max_iters`` for ECOS).

Need to expose MOSEK solver settings. Ones of practical interest
are ``mosek.dparam.intpnt_co_tol_near_rel`` and
``mosek.iparam.intpnt_scaling``.

Send sage_benchmarks primal problem 2, params (p=0, q=3, ell=0, nontriv
X) to MOSEK. With the default scaling, it returns a significantly
infeasible solution.

## Update web documentation to run nosetests, when installing from pip

The command ``nosetests sageopt`` should work.

## Increase compatibility with cvxpy

Make Problem objects take two arguments instead of three;
the objective sense shouldn't be an extra argument.

## Implement the caret (^) operator for Signomial and Polynomial objects

This boils down to implementing ``__xor__``. Add documentation for this
operator, but warn users about the dangers of using it with possible
numeric types.

## Add sageopt.interop, for interfacing with other systems (GPKit)
