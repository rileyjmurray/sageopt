.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614


Polynomials
===========

SAGE relaxations are especially well-suited to sparse polynomials, or polynomials of high
degree. Sageopt provides a symbolic representation for such functions with the
:class:`sageopt.Polynomial` class. The Polynomial class thinks in terms of the following expression:

.. math::

   x \mapsto \sum_{i=1}^m c_i \prod_{j=1}^n {x_j}^{\alpha_{ij}}

i.e. with parameters :math:`\alpha \in R^{m \times n}`, and :math:`c \in R^m`. This
page contains the documentation for this class, as well as discussion on the concept of
:ref:`condsagepolys`.

.. _polyobj:

Polynomial objects
------------------

.. autoclass:: sageopt.symbolic.polynomials.Polynomial
    :members:

.. automethod:: sageopt.symbolic.polynomials.standard_poly_monomials

.. _condsagepolys:

"Conditioning"
--------------

A primary contribution of MCW2019_ was to show that SAGE can naturally handle certain
structured constraints in optimization problems.
The process by which these constraints are handled is known as *partial dualization*.
You can think of partial dualization as a type of "conditioning", in the sense of conditional
probability.

In the polynomial case, the "nice" sets :math:`X` are those satisfying three properties

 1. invariance under reflection about the :math:`n`
    hyperplanes :math:`H_i = \{ (x_1,\ldots,x_n) : x_i = 0 \}`.

 2. the set :math:`X \cap R^n_{++}` is `log-convex <https://arxiv.org/abs/1812.04074>`_, and

 3. the closure of :math:`X \cap R^n_{++}` equals :math:`X \cap R^n_+`

Take a look at Section 4 of `MCW2019 <https://arxiv.org/abs/1907.00814>`_ to see why this is the case.

Sageopt is designed so users can take advantage of partial dualization without being
experts on the subject. Towards this end, sageopt includes a function which can infer
a suitably structured :math:`X` from a given collection of polynomial equations and inequalities.
That function is described below.

.. autofunction:: sageopt.relaxations.sage_polys.conditional_sage_data

The function above captures a small portion of what is possible with conditional SAGE
certificates for polynomials. In order to take more full advantage of the possibilities,
you will need to describe the set yourself. The following class is sageopt's standard
for describing such sets.

.. autoclass:: sageopt.symbolic.polynomials.PolyDomain
    :members:
