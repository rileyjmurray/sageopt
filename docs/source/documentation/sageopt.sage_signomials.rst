.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614

.. _workwithsagesigs:

Working with SAGE signomials.
=============================

SAGE signomials can be used for optimization (i.e. signomial programming) and certifying function nonnegativity.
This page describes core functions which can assist in these goals. These functions are

 - :func:`sageopt.sig_relaxation`,
 - :func:`sageopt.sig_constrained_relaxation`,
 - :func:`sageopt.sig_solrec`,
 - :func:`sageopt.local_refine`,
 - :func:`sageopt.relaxations.sage_sigs.sage_feasibility`, and
 - :func:`sageopt.relaxations.sage_sigs.sage_multiplier_search`.

The functions described here are largely reference
implementations. Depending on the specifics of your problem, it may be beneficial to implement variants of these
functions by directly working with sageopt's backend: coniclifts.
Newcomers to sageopt might benefit from reading this page in
one browser window, and keeping our page of :ref:`allexamples` open in an adjacent window.
It might also be useful to have a copy of MCW2019_ at hand, since that article is
referenced throughout this page.

Optimization
------------

There are two main functions for generating SAGE relaxations of signomial programs: ``sig_relaxation`` and
``sig_constrained_relaxation``. Both of these functions can handle constraints, but they differ in
which constraints they allow. Tractable convex constraints are allowed by both functions, and are
discussed in prerequisite documentation :ref:`condsagesigs`. Solution recovery is addressed after
these functions have been described.

No constraints, or convex constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autofunction:: sageopt.sig_relaxation


When ``form='primal'``, the problem returned by ``sig_relaxation`` can be stated in full generality without too much
trouble. We define a modulator signomial ``t`` (with the canonical choice ``t = Signomial(f.alpha, np.ones(f.m))``),
then return problem data representing

.. math::

      \begin{align*}
         \mathrm{maximize} &~ \gamma \\
         \text{subject to} &~ \mathtt{f{\_}mod} := t^\ell \cdot (f - \gamma), \text{ and} \\
                    &~ \mathtt{f{\_}mod.c} \in C_{\mathrm{SAGE}}(\mathtt{f{\_}mod.alpha}, X)
      \end{align*}

The rationale behind this formation is simple: the minimum value of a function :math:`f` over a set :math:`X` is
equal to the largest number :math:`\gamma` where :math:`f - \gamma` is nonnegative over :math:`X`.
The SAGE constraint in the problem is a proof that :math:`f - \gamma` is nonnegative over :math:`X`.
However the SAGE constraint may be too restrictive, in that it's possible that :math:`f - \gamma` is nonnegative on
:math:`X`,
but not "X-SAGE".
Increasing :math:`\ell` expands the set of functions which SAGE can prove as nonnegative, and thereby
improve the quality the bound produced on :math:`f_X^\star`.
The improved bound comes at the expense of solving a larger optimization problem.
For more discussion, refer to Section 2.3 of MCW2019_.

Arbitrary constraints
~~~~~~~~~~~~~~~~~~~~~

The next function allows the user to specify their problem not only with convex constraints via a set
":math:`X`", but also with explicit signomial inequality constraints and equality constraints.
Such explicit signomial constraints are necessary when the feasible set is nonconvex,
although they can be useful in other contexts.


.. autofunction:: sageopt.sig_constrained_relaxation


Before we move on to solution recovery, we take some time to describe the precise meanings of parameters
``p`` and ``ell`` in ``sig_constrained_relaxation``.
In primal form, ``sig_constrained_relaxation`` operates by moving explicit signomial constraints into a Lagrangian,
and attempting to certify the Lagrangian as nonnegative over ``X``;
this is a standard combination of the concepts reviewed in Section 2 of MCW2019_.
Parameter ``ell`` is essentially the same as in ``sig_relaxation``: to improve the strength of the SAGE
proof system, modulate the Lagrangian ``L - gamma`` by powers of the signomial
``t = Signomial(L.alpha, np.ones(L.m))``.
Parameters ``p`` and ``q`` affect the *unmodulated Lagrangian* seen by ``sig_constrained_relaxation``;
this unmodulated Lagrangian is constructed with the following function.


.. autofunction:: sageopt.relaxations.sage_sigs.make_sig_lagrangian

For detailed discussion on how ``make_sig_lagrangian`` is used to construct primal and dual SAGE relaxations,
refer to :ref:`dualitydesign`.

Solution recovery
~~~~~~~~~~~~~~~~~

Section 3.2 of MCW2019_ introduces two solution recovery algorithms for dual SAGE relaxations.
The main algorithm ("Algorithm 1") is implemented by sageopt's function ``sig_solrec``, and the second algorithm
("Algorithm 1L") is simply to use a local solver to refine the solution produced by the main algorithm.
The exact choice of local solver is not terribly important. For completeness, sageopt includes
a ``local_refine`` function which relies on the COBYLA solver, as described in MCW2019_.

.. autofunction:: sageopt.sig_solrec

``sig_solrec`` actually implements a slight generalization of "Algorithm 1" from MCW2019_. The generalization
is used to improve performance in more complex SAGE relaxations, such as those from
``sig_constrained_relaxation`` with ``ell > 0``.

Users can replicate "Algorithm 1L" from MCW2019_ by running ``sig_solrec``, and then applying the following function
to its output.

.. autofunction:: sageopt.local_refine

Nonnegativity
-------------

Sageopt offers two pre-made functions for certifying nonnegativity and finding SAGE decompositions:
``sage_feasibility`` and ``sage_multiplier_search``. These functions are accessible as top-level imports
``sageopt.sage_feasibility`` and ``sageopt.sage_multiplier_search``, where they accept Signomial or Polynomial objects.
The documentation below addresses the Signomial implementations.

.. autofunction:: sageopt.relaxations.sage_sigs.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_sigs.sage_multiplier_search

