This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Add unittests for minimax-free relaxations of constrained signomial programs

Right now, all system-level tests of the relaxations package use unconstrained
relaxations, or the mixed conditional-SAGE / Lagrangian relaxations associated
with sig_constrained_relaxation. There should be some dedicated tests for
sig_relaxation with conditioning.


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

Need to espose MOSEK solver settings. Ones of practical interest
are ``mosek.dparam.intpnt_co_tol_near_rel`` and
``mosek.iparam.intpnt_scaling``.

Send sage_benchmarks primal problem 2, params (p=0, q=3, ell=0, nontriv
X) to MOSEK. With the default scaling, it returns a significantly
infeasible solution.