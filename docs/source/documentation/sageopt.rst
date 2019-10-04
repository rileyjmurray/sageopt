Overview of sageopt
===================

Sageopt provides functionality for optimization and checking nonnegativity of signomials and polynomials.
This functionality is built around simple classes for Signomial and Polynomial objects, and a custom-built
backend for interfacing with convex optimization solvers. We imagine that most users are interested in *either*
signomials *or* polynomials, and so the documentation in these cases is given separately.

.. toctree::
   :maxdepth: 3

   Signomials <sageopt.signomials>
   Polynomials <sageopt.polynomials>


If the ``relaxations`` subpackage does not contain some functionality which you need, then it should be possible to
implement that functionality using ``sageopt.coniclifts`` as a basic, high-level modeling interface.

.. toctree::
   :maxdepth: 2

   Coniclifts <sageopt.coniclifts>


Package structure
-----------------

Sageopt version 0.4 is divided into three subpackages:
 1. ``sageopt.symbolic``,
 2. ``sageopt.relaxations``, and
 3. ``sageopt.coniclifts``.
