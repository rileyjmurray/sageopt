.. _nonneg:

=============
Nonnegativity
=============

Although most of sageopt is designed with optimization in mind, the underlying mechanism of "SAGE certificates"
is really about certifying function nonnegativity. Certifying function nonnegativity (whether globally over
:math:`\mathbb{R}^n`, or over some proper subset :math:`X \subset \mathbb{R}^n`) is an important problem in
the mathematical field of algebraic geometry, and has practical applications to control theory. This page discusses
how you can use sageopt to certify signomial and polynomial nonnegativity.

pre-made functions
------------------

Sageopt offers two pre-made functions for certifying nonnegativity and finding SAGE decompositions:
``sage_feasibility`` and ``sage_multiplier_search``. These functions are accessible as top-level imports
``sageopt.sage_feasibility`` and ``sageopt.sage_multiplier_search``, where they accept Signomial or Polynomial objects.

.. autofunction:: sageopt.sage_feasibility

.. autofunction:: sageopt.sage_multiplier_search
