.. _dualitydesign:

Duality in ``sageopt.relaxations``.
===================================


Symbolic duality
----------------

Both ``sig_relaxation`` and ``sig_constrained_relaxation`` allow the user to specify whether they want
primal-form or
dual-form SAGE relaxations. Generally speaking
#. Dual-forms are more useful than primal-forms, and
#. Primal-forms are easier to state than dual-forms.
Dual relaxations are generally more useful, because sageopt includes functions to help recover solutions from a dual
relaxation.
The main purpose of solving the primal relaxation is to verify that reportedly "optimal" primal and dual objectives
are close to one another.
It is a good idea to check this manually, since numerical solvers (such as MOSEK or ECOS) can sometimes
report "optimal" status codes even when a returned solution is infeasible or highly-suboptimal.