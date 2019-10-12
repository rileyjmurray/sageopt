This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Change presolve behavior for trivial AGE cones

DONE: Only presolve-away the trivial AGE cones if the user asks for it.
(Since it *really* slows down problem construction.)

DONE: Update unittests so that code path is still tested with ECOS.
Make sure ECOS can still solve the resulting problems (since I expect
them to have worse conditioning).

TODO: Update rst files and web documentation.

## Add unittests for minimax-free relaxations of constrained signomial programs

Right now, all system-level tests of the relaxations package use unconstrained
relaxations, or the mixed conditional-SAGE / Lagrangian relaxations associated
with sig_constrained_relaxation. There should be some dedicated tests for
sig_relaxation with conditioning.

## Add more tests for polynomial solution recovery

Local_refine_polys_from_sigs could use an isolated unittest.

poly_solrec needs a test for the codepath where ...
 - we use ordinary SAGE cones,
 - linear system sign pattern recovery fails, and moves to greedy_weighted_cut_negatives
 - some components of the moment vector are zero.


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
