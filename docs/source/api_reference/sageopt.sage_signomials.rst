.. _workwithsagesigs:

Working with SAGE signomials.
=============================


optimization
------------

There are two main functions for generating SAGE relaxations of signomial programs.

The simpler function is called ``sig_relaxation``, it applies to unconstrained problems, or
constrained problems where the feasible set :math:`X` has a tractable convex representation.

The more complicated function is called ``sig_constrained_relaxation``, it applies when the problem
includes explicit signomial inequality constraints (:math:`g(x) \geq 0`) or equality constraints (:math:`g(x) = 0`).
Explicit signomial constraints are necessary when the feasible set is nonconvex,
although explicit constraints can sometimes be useful even when the feasible set is convex.

These functions allow the user to specify whether they want to construct primal-form or dual-form SAGE
relaxations. Generally speaking, the dual is more useful, because sageopt includes additional
functions to help recover solutions from a dual relaxation. Primal relaxations have
important theoretical properties, but we will not describe those here.
From a practical standpoint, the main purpose of constructing and solving the primal relaxation is to verify that
primal and dual objectives are close to one another. It is a good idea to check this manually, since numerical
solvers (such as MOSEK, or ECOS, or SCS) can sometimes report "optimal" status codes even when a returned solution is
infeasible or highly-suboptimal.


.. autofunction:: sageopt.sig_relaxation

.. autofunction:: sageopt.sig_constrained_relaxation

.. autofunction:: sageopt.relaxations.sage_sigs.make_sig_lagrangian


certificates of nonnegativity
-----------------------------

.. autofunction:: sageopt.relaxations.sage_sigs.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_sigs.sage_multiplier_search


helper functions
----------------

.. autofunction:: sageopt.relaxations.sage_sigs.conditional_sage_data

