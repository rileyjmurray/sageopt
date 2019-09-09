Overview of sageopt's structure
===============================

Sageopt version 0.3 is divided into three subpackages:
 1. ``sageopt.symbolic``,
 2. ``sageopt.relaxations``, and
 3. ``sageopt.coniclifts``.

Most users will first want to read documentation for ``sageopt.symbolic``, because this tells you how to construct
and use ``Signomial`` and ``Polynomial`` objects which you need to define your optimization problems.

Once you know how to instantiate ``Signomial`` and ``Polynomial`` objects, you will want to use the ``sageopt.relaxations`` subpackage to manage building and solving a SAGE relaxation for your optimization problem.

If the ``relaxations`` subpackage does not contain some functionality which you need, then it should be possible to
implement that functionality using ``sageopt.coniclifts`` as a basic, high-level modeling interface.

.. toctree::
   :maxdepth: 2

   Signomial and Polynomial objects <sageopt.symbolic>
   SAGE signomials <sageopt.sage_signomials>
   SAGE polynomials <sageopt.sage_polynomials>
   Coniclifts <sageopt.coniclifts>
