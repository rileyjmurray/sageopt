.. _workwithsagepolys:

Working with SAGE polynomials.
==============================

There are two main functions for generating SAGE relaxations of polynomial optimization problems: ``poly_relaxation``
and ``poly_constrained_relaxation``. Both of these functions can handle constraints, but they differ in
which constraints they allow.

``poly_relaxation`` requires that all constraints are incorporated into
a set :math:`X` , where the pointwise elementwise absolute value :math:`|X|` is a `log-log convex set
<https://arxiv.org/abs/1812.04074>`_. For details on what sets :math:`X` are allowed, refer to Section 4.1 of
`MCW2019 <https://arxiv.org/abs/1907.00814>`_.
In more general settings, you can use ``poly_constrained_relaxation``. In
addition to allowing sets :math:`X` described above, this function
allows explicit polynomial inequality constraints
(:math:`g(x) \geq 0`) and equality constraints (:math:`g(x) = 0`).

Both ``poly_relaxation`` and ``poly_constrained_relaxation`` allow the user to specify whether they want primal-form or
dual-form SAGE
relaxations. Generally speaking, the dual is more useful. This is because sageopt includes additional
functions to help recover solutions from a dual relaxation.
Primal relaxations have important theoretical properties, but we will not describe those here.
From a practical standpoint, the main purpose of solving the primal relaxation is to verify that reportedly "optimal"
primal and dual objectives are close to one another.
It is a good idea to check this manually, since numerical solvers (such as MOSEK or ECOS) can sometimes
report "optimal" status codes even when a returned solution is infeasible or highly-suboptimal.


Optimization
------------

.. autofunction:: sageopt.poly_relaxation

.. autofunction:: sageopt.poly_constrained_relaxation


The documentation for ``poly_constrained_relaxation`` concerning parameters ``p``, ``q``, and ``ell`` (typeset in
math as :math:`(p, q, \ell)`) is admittedly somewhat vague. There are two places you can look for the precise
meanings of these parameters: Section 4.4
of the article `MCW2019 <https://arxiv.org/abs/1907.00814>`_, and appropriate helper functions in sageopt's source code.
Many readers will find it difficult to understand Section 4.4 of
`MCW2019 <https://arxiv.org/abs/1907.00814>`_ without first reading all the article's Section 2 (which is
deliberately written as background for non-experts). When it comes to souce code, the best place to look for
parameters ``p`` and ``q`` is the function ``make_poly_lagrangian``, which we describe below.


.. autofunction:: sageopt.relaxations.sage_polys.make_poly_lagrangian


Certificates of nonnegativity
-----------------------------

.. autofunction:: sageopt.relaxations.sage_polys.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_polys.sage_multiplier_search


Helper functions
----------------

.. autofunction:: sageopt.relaxations.sage_polys.conditional_sage_data

