.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _workwithsagepolys:

Working with SAGE polynomials.
==============================

SAGE polynomials can be used for optimization, and certifying function nonnegativity.
This page describes core functions which can assist in these goals. These functions are

 - :func:`sageopt.poly_relaxation`,
 - :func:`sageopt.poly_constrained_relaxation`,
 - :func:`sageopt.poly_solrec`,
 - :func:`sageopt.local_refine_polys_from_sigs`,
 - :func:`sageopt.relaxations.sage_polys.sage_feasibility`, and
 - :func:`sageopt.relaxations.sage_polys.sage_multiplier_search`.

The functions described here are largely reference
implementations. Depending on the specifics of your problem, it may be beneficial to implement variants of these
functions by directly working with sageopt's backend: coniclifts.
Newcomers to sageopt might benefit from reading this page in
one browser window, and keeping our page of :ref:`allexamples` open in an adjacent window.
It might also be useful to have a copy of MCW2019_ at hand, since that article is
referenced throughout this page.

Optimization
------------

There are two main functions for generating SAGE relaxations of polynomial optimization problems: ``poly_relaxation``
and ``poly_constrained_relaxation``. Both of these functions can handle constraints, but they differ in
which constraints they allow. This section presents these functions and then addresses solution recovery.

Structured constraints only
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``poly_relaxation`` requires that any and all constraints are incorporated into
a set :math:`X` , which satisfies the properties for polynomial :ref:`condsagepolys`.

.. autofunction:: sageopt.poly_relaxation

General constraints
~~~~~~~~~~~~~~~~~~~

In more general settings, you can use ``poly_constrained_relaxation``. In
addition to allowing sets :math:`X` described above, this function
allows explicit polynomial inequality constraints
(:math:`g(x) \geq 0`) and equality constraints (:math:`g(x) = 0`).

.. autofunction:: sageopt.poly_constrained_relaxation

Before we move on to solution recovery, we take some time to describe the precise meanings of parameters
``p`` and ``ell`` in ``poly_constrained_relaxation``.
In primal form, ``poly_constrained_relaxation`` operates by moving explicit polynomial constraints into a Lagrangian,
and attempting to certify the Lagrangian as nonnegative over ``X``;
this is a standard combination of the concepts reviewed in Section 2 of MCW2019_.
Parameter ``ell`` is essentially the same as in ``poly_relaxation``: to improve the strength of the SAGE
proof system, modulate the Lagrangian ``L - gamma`` by powers of the polynomial
``t = Polynomial(2 * L.alpha, np.ones(L.m))``.
Parameters ``p`` and ``q`` affect the *unmodulated Lagrangian* seen by ``poly_constrained_relaxation``;
this unmodulated Lagrangian is constructed with the following function.

.. autofunction:: sageopt.relaxations.sage_polys.make_poly_lagrangian

Solution recovery
~~~~~~~~~~~~~~~~~

Section 4.2 of MCW2019_ introduces two solution recovery algorithms for dual SAGE relaxations.
The main algorithm ("Algorithm 2") is implemented by sageopt's function ``poly_solrec``, and the second algorithm
("Algorithm 2L") is simply to use a local solver to refine the solution produced by the main algorithm.
The exact choice of local solver is not terribly important. For completeness, sageopt includes
a generic :func:`sageopt.local_refine` function which relies on the COBYLA solver.

.. autofunction:: sageopt.poly_solrec

When faced with a polynomial optimization problem over nonnegative variables,
one should formulate the problem in terms of signomials.
This reformulation is without loss of generality from the perspective of solving a SAGE relaxation,
but the local-refinement stage of solution recovery is somewhat different.
The following function may be important if a polynomial optimization problem has
some variable equal to zero in an optimal solution.

.. autofunction:: sageopt.local_refine_polys_from_sigs


Nonnegativity
-------------

Sageopt offers two pre-made functions for certifying nonnegativity and finding SAGE decompositions:
``sage_feasibility`` and ``sage_multiplier_search``. These functions are accessible as top-level imports
``sageopt.sage_feasibility`` and ``sageopt.sage_multiplier_search``, where they accept Signomial or Polynomial objects.
The documentation below addresses the Signomial implementations.

.. autofunction:: sageopt.relaxations.sage_polys.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_polys.sage_multiplier_search

