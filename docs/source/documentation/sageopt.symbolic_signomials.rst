.. _MCW2019: https://arxiv.org/abs/1907.00814

Signomials
==========

A signomial is a linear combination of exponentials, composed with linear functions.
Signomials look like the following:

.. math::

   x \mapsto \sum_{i=1}^m c_i \exp({\alpha}_i \cdot x)

The class :class:`sageopt.Signomial` implements a symbolic representation of such functions.
The section :ref:`sigobj` covers this class, and an extra helper function
for constructing signomials.

When signomials are considered in exponential form -- as is *always* done in sageopt --
they have a powerful connection to convexity. This connection is very important for
constrained optimization, and is described in the section :ref:`condsagesigs`.

.. _sigobj:

Signomial objects
-----------------

This section covers the Signomial class, and the function
:func:`sageopt.standard_sig_monomials`. The helper function
is very convenient for constructing Signomial objects; you
might even use it more than the Signomial constructor itself.
Nevertheless, it is a good idea to review the Signomial class first.

.. autoclass:: sageopt.Signomial
    :members:

.. automethod:: sageopt.symbolic.signomials.standard_sig_monomials

.. _condsagesigs:

"Conditioning"
--------------

The primary contribution of MCW2019_ was to show that convex sets have a special place
in the theory of SAGE relaxations.
In particular, SAGE can incorporate convex constraints into a problem by a lossless process
known as *partial dualization*.
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

.. autofunction:: sageopt.relaxations.sage_sigs.conditional_sage_data

It is possible that the function above cannot capture a convex set of interest. This is
particularly likely if the desired convex set is not naturally described
by signomials. In such situations, you will need to
interact with the SigDomain class directly.


.. autoclass:: sageopt.SigDomain
    :members:
