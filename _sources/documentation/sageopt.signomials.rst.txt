.. _MCW2019: https://arxiv.org/abs/1907.00814

Signomials
==========

A signomial is a linear combination of exponentials, composed with linear functions.
Signomials look like the following:

.. math::

   x \mapsto \sum_{i=1}^m c_i \exp({\alpha}_i \cdot x)

:ref:`Signomial objects <sigobj>` covers sageopt's Signomial class, plus two extra helper functions.
The section on :ref:`conditioning<condsagesigs>` covers the basics of a powerful connection between
signomials and convexity.
Sageopt has convenience functions for constructing and working with convex relaxations of signomial
minimization problems (both constrained, and unconstrained).
Those convenience functions are described in the section on :ref:`optimization<workwithsagesigs>`.
We also address some :ref:`advanced topics<advancedsigs>`.

.. _sigobj:

Signomial objects
-----------------

This section covers how to construct and use instances of the :class:`sageopt.Signomial` class.

.. autoclass:: sageopt.Signomial
    :members:

.. autofunction:: sageopt.standard_sig_monomials

.. _condsagesigs:

Conditioning
------------

Convex sets have a special place in the theory of SAGE relaxations.
In particular, SAGE can incorporate convex constraints into a problem by a lossless process
known as *partial dualization* MCW2019_.
You can think of partial dualization as a type of "conditioning", in the sense of conditional
probability.

We designed sageopt so users can leverage the full power of partial dualization without being
experts on the subject.
If you want to optimize a signomial over the set

.. math::

    \Omega = \{ x \,:\, g(x) \geq 0 \text{ for }g \in \mathtt{gts}, ~~ \phi(x)=0 \text{ for } \phi \in \mathtt{eqs}\}

then you just need to focus on constructing the lists of signomials ``gts`` and ``eqs``.
Once these lists are constructed, you can call the following function to obtain
a convex set :math:`X \supset \Omega` which is implied by the constraint signomials.

.. autofunction:: sageopt.relaxations.sage_sigs.infer_domain

It is possible that the function above cannot capture a convex set of interest. This is
particularly likely if the desired convex set is not naturally described
by signomials. If your find yourself in this situation, refer to the :ref:`advanced topics<advancedsigs>` section.

.. _workwithsagesigs:

Optimization
------------

Here are sageopt's convenience functions for signomial optimization:

 - :func:`sageopt.sig_relaxation`,
 - :func:`sageopt.sig_constrained_relaxation`,
 - :func:`sageopt.sig_solrec`, and
 - :func:`sageopt.local_refine`.

We assume the user has already read the section on signomial :ref:`condsagesigs`.
Newcomers to sageopt might benefit from reading this section in
one browser window, and keeping our page of :ref:`allexamples` open in an adjacent window.
It might also be useful to have a copy of MCW2019_ at hand, since that article is
referenced throughout this section.

A remark: The functions described here are *reference implementations*.
Depending on the specifics of your problem, it may be beneficial to implement variants of these
functions by directly working with sageopt's backend: :ref:`coniclifts<coniclifts>`.

Optimization with convex constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

Optimization with arbitrary signomial constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next function allows the user to specify their problem not only with convex constraints via a set
":math:`X`", but also with explicit signomial equations and inequalities.
Such signomial constraints are necessary when the feasible set is nonconvex,
although they can be useful in other contexts.


.. autofunction:: sageopt.sig_constrained_relaxation

For further explanation of the parameters ``p``, ``q``, and ``ell`` in the function above, we refer the user
to the :ref:`advancedsigs` section.


Solution recovery for signomial optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


.. _advancedsigs:

Advanced topics
---------------

SigDomain objects
~~~~~~~~~~~~~~~~~

.. autoclass:: sageopt.SigDomain
    :members:

reference hierarchy parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we describe the precise meanings of parameters
``p`` and ``ell`` in :func:`sig_constrained_relaxation<sageopt.sig_constrained_relaxation>`.
In primal form, :func:`sig_constrained_relaxation<sageopt.sig_constrained_relaxation>` operates by
moving explicit signomial constraints into a Lagrangian,
and attempting to certify the Lagrangian as nonnegative over ``X``;
this is a standard combination of the concepts reviewed in Section 2 of MCW2019_.
Parameter ``ell`` is essentially the same as in :func:`sig_relaxation<sageopt.sig_relaxation>`:
to improve the strength of the SAGE
proof system, modulate the Lagrangian ``L - gamma`` by powers of the signomial
``t = Signomial(L.alpha, np.ones(L.m))``.
Parameters ``p`` and ``q`` affect the *unmodulated Lagrangian* seen by
:func:`sig_constrained_relaxation<sageopt.sig_constrained_relaxation>`;
this unmodulated Lagrangian is constructed with the following function.

.. autofunction:: sageopt.relaxations.sage_sigs.make_sig_lagrangian

