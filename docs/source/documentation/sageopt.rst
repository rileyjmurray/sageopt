.. _MCW2019: https://arxiv.org/abs/1907.00814

Overview of sageopt
===================

Sageopt includes functionality for optimization and checking nonnegativity of signomials and polynomials.
The conceptual components here are:

 1. Some classes representing signomials and polynomials, as well as domains to which these functions
    are restricted.

 2. A high-level modeling interface (called "coniclifts") for convex optimization problems.
    This interface includes primal and dual "SAGE constraints."

Sageopt then takes these two components, and defines various functions to reduce user's barrier to
entry in working with SAGE relaxations.

Our main concern is to make it easier to get started with *optimization*.
SAGE relaxations for optimization have different "primal" and "dual" forms, and it would take a bit of
expertise to implement the advanced dual-form relaxations from scratch.
The documentation below will bring you up to speed on the basics of sageopt's signomial
and polynomial optimization features, as well as the most crucial prerequisites.

.. toctree::
   :maxdepth: 2

   Signomials <sageopt.signomials>
   Polynomials <sageopt.polynomials>

At some point, users will need to interact with sageopt's "coniclifts" backend.
This is especially true if you intend to use sageopt mostly for certifying function nonnegativity.
Documentation for coniclifts is given below.

.. toctree::
   :maxdepth: 2

   Coniclifts <sageopt.coniclifts>
   Nonnegativity <sageopt.nonnegativity>

Finally, sageopt provides some basic features for interacting with optimization packages
in the python ecosystem: GPKit and CVXPY.