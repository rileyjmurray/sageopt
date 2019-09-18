Overview of sageopt's structure
===============================

Sageopt version 0.3 is divided into three subpackages:
 1. ``sageopt.symbolic``,
 2. ``sageopt.relaxations``, and
 3. ``sageopt.coniclifts``.

Most users will first want to read documentation for ``sageopt.symbolic``. This tells you how to construct
and use ``Signomial`` and ``Polynomial`` objects, which you will need to define optimization problems or
nonnegativity problems.

.. toctree::
   :maxdepth: 2

   Signomials <sageopt.symbolic_signomials>
   Polynomials <sageopt.symbolic_polynomials>

Once you know how to instantiate ``Signomial`` and ``Polynomial`` objects, you will want to use the
``sageopt.relaxations`` subpackage to manage building a SAGE relaxation, solving that SAGE relaxation, and
working with the solution.

.. toctree::
   :maxdepth: 3

   SAGE for signomials <sageopt.sage_signomials>
   SAGE for polynomials <sageopt.sage_polynomials>
   Notes on duality <sageopt.relaxations_duality>

If the ``relaxations`` subpackage does not contain some functionality which you need, then it should be possible to
implement that functionality using ``sageopt.coniclifts`` as a basic, high-level modeling interface.

.. toctree::
   :maxdepth: 2

   Coniclifts <sageopt.coniclifts>
