.. _MCW2019: https://arxiv.org/abs/1907.00814

Signomials
==========

This page describes two classes, and two helper functions

 - :class:`sageopt.Signomial`,
 - :class:`sageopt.SigDomain`,
 - :func:`sageopt.symbolic.signomials.standard_sig_monomials`, and
 - :func:`sageopt.relaxations.sage_sigs.conditional_sage_data`.

It is possible to access simplest, bare-bones functionality of sageopt by only knowing
how to use the helper function ``standard_sig_monomials``. If you plan on solving problems
in a programmatic way (i.e. not typing out every signomial by hand), then you will
need to know the basics of the ``Signomial`` class. The Signomial class is pretty simple,
so we actually recommend all users read the associated documentation at least once.


The ``SigDomain`` class provides an abstraction needed for "conditional SAGE relaxations",
which are very important for constrained optimization.
The helper function ``conditional_sage_data`` is the primary way to get your hands on
a SigDomain.
There is a good chance you will never need to declare a SigDomain object on your own,
although this page does discuss such advanced usage.

.. automethod:: sageopt.symbolic.signomials.standard_sig_monomials


Signomial objects
-----------------

.. autoclass:: sageopt.Signomial
    :members:


.. _condsagesigs:

"Conditioning"
--------------

The primary contribution of MCW2019_ was to show that convex sets have a special place
in the theory of SAGE relaxations.
In particular, SAGE can incorporate convex constraints into a problem by a lossless process
known as *partial dualization*.
You can think of partial dualization as a type of "conditioning", in the sense of "conditional
probability".

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
