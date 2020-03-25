.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614


Polynomials
===========

SAGE relaxations are well-suited to sparse polynomials, or polynomials of high
degree. Sageopt provides a symbolic representation for such functions with the
:class:`sageopt.Polynomial` class. The Polynomial class thinks in terms of the following expression:

.. math::

   x \mapsto \sum_{i=1}^m c_i \prod_{j=1}^n {x_j}^{\alpha_{ij}}

i.e. with parameters :math:`\alpha \in \mathbb{N}^{m \times n}`, and :math:`c \in \mathbb{R}^m`.
This page contains (1) the documentation for this class, (2) discussion on the concept of
:ref:`conditioning<condsagepolys>`, (3) documentation for pre-built functions which assist in polynomial
:ref:`optimization<workwithsagepolys>`, and (4) some :ref:`advanced topics<advancedpolys>`.

.. _polyobj:

Polynomial objects
------------------

.. autoclass:: sageopt.symbolic.polynomials.Polynomial
    :members:

.. automethod:: sageopt.symbolic.polynomials.standard_poly_monomials

.. _condsagepolys:

Conditioning
------------

SAGE can naturally handle certain
structured constraints in optimization problems, or nonnegativity problems.
The process by which these constraints are handled is known as *partial dualization*.
You can think of partial dualization as a type of "conditioning", in the sense of conditional
probability.

In the polynomial case, the "nice" sets :math:`X` are those satisfying three properties

 1. invariance under reflection about the :math:`n`
    hyperplanes :math:`H_i = \{ (x_1,\ldots,x_n) : x_i = 0 \}`.

 2. the set :math:`X \cap \mathbb{R}^n_{++}` is `log-convex <https://arxiv.org/abs/1812.04074>`_, and

 3. the closure of :math:`X \cap \mathbb{R}^n_{++}` equals :math:`X \cap \mathbb{R}^n_+`

Take a look at Section 4 of MCW2019_ to see why this is the case.

Sageopt is designed so users can take advantage of partial dualization without being
experts on the subject. Towards this end, sageopt includes a function which can infer
a suitably structured :math:`X` from a given collection of polynomial equations and inequalities.
That function is described below.

.. autofunction:: sageopt.relaxations.sage_polys.infer_domain

The function above captures a small portion of what is possible with conditional SAGE
certificates for polynomials. In order to take more full advantage of the possibilities,
you will need to describe the set yourself. Refer to the :ref:`advancedpolys` section
for more information.

.. _workwithsagepolys:

Optimization
------------

Here are sageopt's convenience functions for polynomial optimization:

 - :func:`sageopt.poly_relaxation`,
 - :func:`sageopt.poly_constrained_relaxation`,
 - :func:`sageopt.poly_solrec`, and
 - :func:`sageopt.local_refine_polys_from_sigs`.

We assume the user has already read the section on polynomial :ref:`conditioning<condsagepolys>`.
Newcomers to sageopt might benefit from reading this page in
one browser window, and keeping our page of :ref:`allexamples` open in an adjacent window.
It might also be useful to have a copy of MCW2019_ at hand, since that article is
referenced throughout this section.

A remark: The functions described here are *reference implementations*.
Significant speed improvements are possible if you build variants of these functions
directly with sageopt's backend: :ref:`coniclifts`.

Optimization with structured constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``poly_relaxation`` requires that any and all constraints are incorporated into
a set :math:`X` , which satisfies the properties for polynomial :ref:`conditioning<condsagepolys>`.

.. autofunction:: sageopt.poly_relaxation

Optimization with arbitrary polynomial constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In more general settings, you can use ``poly_constrained_relaxation``. In
addition to allowing sets :math:`X` described above, this function
allows explicit polynomial inequality constraints
(:math:`g(x) \geq 0`) and equality constraints (:math:`g(x) = 0`).

.. autofunction:: sageopt.poly_constrained_relaxation

For further explanation of the parameters ``p``, ``q``, and ``ell`` in the function above, we refer the user
to the :ref:`advanced topics<advancedpolys>` section.

Solution recovery for polynomial optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



.. _advancedpolys:

Advanced topics
---------------

PolyDomain objects
~~~~~~~~~~~~~~~~~~

.. autoclass:: sageopt.symbolic.polynomials.PolyDomain
    :members:

Reference hierarchy parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we describe the precise meanings of parameters
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