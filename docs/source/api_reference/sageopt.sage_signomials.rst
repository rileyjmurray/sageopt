.. _workwithsagesigs:

Working with SAGE signomials.
=============================


Optimization
------------

There are two main functions for generating SAGE relaxations of signomial programs: ``sig_relaxation`` and
``sig_constrained_relaxation``. Both of these functions can handle constraints, but they differ in
which constraints they allow. ``sig_relaxation`` requires that all constraints are incorporated into
a set :math:`X`, which has a tractable convex representation. In more general settings,
you can use ``sig_constrained_relaxation``. In addition to convex constraints represented by :math:`X`, this function
allows explicit signomial inequality constraints
(:math:`g(x) \geq 0`) and equality constraints (:math:`g(x) = 0`).
Explicit signomial constraints are necessary when the feasible set is nonconvex,
although they can be useful in other contexts.

Both ``sig_relaxation`` and ``sig_constrained_relaxation`` allow the user to specify whether they want primal-form or
dual-form SAGE
relaxations. Generally speaking, the dual is more useful. This is because sageopt includes additional
functions to help recover solutions from a dual relaxation.
Primal relaxations have important theoretical properties, but we will not describe those here.
From a practical standpoint, the main purpose of solving the primal relaxation is to verify that reportedly "optimal"
primal and dual objectives are close to one another.
It is a good idea to check this manually, since numerical solvers (such as MOSEK, or ECOS, or SCS) can sometimes
report "optimal" status codes even when a returned solution is infeasible or highly-suboptimal.


.. autofunction:: sageopt.sig_relaxation

.. autofunction:: sageopt.sig_constrained_relaxation

.. autofunction:: sageopt.relaxations.sage_sigs.make_sig_lagrangian


Certificates of nonnegativity
-----------------------------

.. autofunction:: sageopt.relaxations.sage_sigs.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_sigs.sage_multiplier_search


Helper functions
----------------

.. autofunction:: sageopt.relaxations.sage_sigs.conditional_sage_data

