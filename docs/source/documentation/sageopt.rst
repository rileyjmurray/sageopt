.. _MCW2019: https://arxiv.org/abs/1907.00814

Overview of sageopt
===================

Sageopt includes functionality for optimization and checking nonnegativity of signomials and polynomials.
The conceptual components here are:

 1. Some classes representing signomials, polynomials, and domains over which these functions
    are defined.

 2. A high-level modeling interface (called "coniclifts") for convex optimization problems.
    Advanced users can interact with native primal or dual "SAGE constraints" via this interface.

Sageopt then takes these two components, and defines various functions to reduce user's barrier to
entry in working with SAGE relaxations. Our primary concern is to make it easier to use SAGE relaxations
for *optimization*; those problems have different "primal" and "dual" forms, and it can take a bit of
expertise to implement the more advanced dual-form relaxations.

The documentation below will bring you up to speed on the basics of sageopt's signomial
and polynomial optimization features, as well as the most crucial prerequisites.

.. toctree::
   :maxdepth: 3

   Signomials <sageopt.signomials>
   Polynomials <sageopt.polynomials>

At some point, users will need to interact with sageopt's "coniclifts" backend.
This is especially true if you intend to use sageopt mostly for certifying function nonnegativity
(although we do have a few pre-built functions there as well).
Documentation for coniclifts is given below.

.. toctree::
   :maxdepth: 2

   Coniclifts <sageopt.coniclifts>
   Nonnegativity <sageopt.nonnegativity>
