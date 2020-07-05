
Release History
===============

The notes here are a summary from Sageopt's changelog. Each item in the lists that follow is prefaced by
a term like "coniclifts", "symbolic", or "relaxations"; these terms indicate the primary subpackage of sageopt which
was affected.

[0.5.2] - 2020-07-05
--------------------
Changed
 - coniclifts: heavily refactor precompiled affine atoms.
 - relaxations: dual SAGE relaxations no longer introduce high-level slack variables
   by default.

Added
 - coniclifts: the mosek interface now automatically performs low-level dualization for certain problems.
 - coniclifts: ``prob.solve`` accepts an argument to override or force dualization (when using MOSEK).


[0.5.1] - 2020-06-21
--------------------
Removed
 - coniclifts: compile_linear_expression.

Changed
 - symbolic: signomials now broadcast in certain arithmetic with ndarrays.

Added
 - coniclifts: "abs" and "pos" NonlinearScalarAtoms.
 - coniclifts: a field "age_witnesses" to ``PrimalSageCone`` objects.
 - coniclifts: ``compile_objective``.
 - coniclifts: an option to eliminate all equality constraints when compiling a ``PrimalSageCone``.
 - coniclifts: ``PrimalSageCone`` violation now computed by evaluating the support function of ``X``.
 - symbolic: SigDomain objects have a ``suppfunc`` method, to evaluate support function.


[0.5.0] - 2020-03-25
--------------------
Removed
 - coniclifts: private compilation constants (e.g. ``_REDUCTION_SOLVER_``) in ``sage_cones.py``
 - coniclifts: private solver parameters (e.g. ``_DEACTIVATE_SCALING``) in ``mosek.py``

Changed
 - symbolic: Signomials are now represented primarily by arrays, rather than a dictionary.
 - symbolic: Signomial and Polynomial constructors: both parameters ``alpha`` and ``c`` are *required*.

Added
 - symbolic: ``Signomial.from_dict`` and ``Polynomial.from_dict`` convenience functions for construction.
 - relaxations: keyword argument checking in ``sig_constrained_relaxation`` and ``poly_constrained_relaxation``.
 - coniclifts: an option to reduce epigraph variable usage in dual SAGE relaxations.
 - coniclifts: mixed-integer support for MOSEK.
 - coniclifts: ``PrimalSageCone.settings`` and ``DualSageCone.settings``: dictionary-valued properties which record
   compilation settings (e.g. pre-solve behavior)
 - coniclifts: Properly expose all MOSEK solver options.
 - interop: Proper tests for GPKit integration.
 - interop: CVXPY integration.


[0.4.2] - 2019-11-18
--------------------
Changed
 - symbolic: ``Signomial.partial`` and ``Polynomial.partial`` to correctly work with nonconstant coefficients.
 - symbolic: Essentially all "attributes" of Signomial or Polynomial objects have become "properties".

Added
 - sageopt: ``sageopt.interop`` a subpackage for interacting with other python projects.
 - symbolic: A ``metadata`` field to Signomial and Polynomial objects.


[0.4.1] - 2019-10-13
--------------------
Removed
 - coniclifts: ``Variable.set_scalar_variables``.
 - coniclifts: The ``name`` field of the ScalarVariable class.
 - coniclifts: Some unused functions in ``coniclifts.base.ScalarAtom``.

Changed
 - symbolic: Fixed a bug in SigDomain and PolyDomain.
 - relaxations: Fixed a bug in the function ``clcons_from_standard_gprep``.
 - coniclifts: ``Variable.value`` is now a settable property.

Added
 - coniclifts: An ``index`` field to the ScalarVariable class.
 - coniclifts: A ``_DEACTIVATE_SCALING_`` static attribute in the ``Mosek`` class.


[0.4.0] - 2019-10-06
--------------------
Removed
 - sageopt: sig_primal, sig_dual, poly_primal, poly_dual (and the four constrained variations thereof)
   as top-level imports within sageopt. These should have been removed in v0.3.4.
 - coniclifts: conditional_sage_cone.py
 - coniclifts: sage_cone.py.

Changed
 - relaxations: ``conditional_sage_data`` is now ``infer_domain``.
 - relaxations: The signature of ``sig_relaxation`` and ``sig_constrained_relaxation``.
 - relaxations: Fixed a bug in ``conditional_sage_data`` (now, ``infer_domain``). Equality constraints were being
   incorrectly compiled into weaker forms.
 - coniclifts: Conditional SAGE constraints now assume that the "conditioning" is feasible.
 - coniclifts: The ``var_name_to_locs`` dict from ``coniclifts.compilers.compile_constrained_system``.
 - coniclifts: Several fields in the ``coniclifts.Problem`` class.
 - coniclifts: How ``ElementwiseConstraint`` objects report their variables.

Added
 - symbolic: SigDomain and PolyDomain classes, to represent "X" with conditional SAGE cones.
 - coniclifts: sage_cones.py.
 - coniclifts: the option to eliminate trivial AGE cones from SAGE relaxations.
 - coniclifts: the option to allow AGE vectors to sum <= c, or == c. The default is <= c.


[0.3.4] - 2019-09-09
--------------------
Changed
 - coniclifts: Major changes to the compilation process. Refer to commit 7cb07866e55c6618ce17d090d52281512ea2351f.

Added
 - relaxations: sig_relaxation and sig_constrained_relaxation (and variants for polynomial problems).


[0.3.3] - 2019-08-10
--------------------
Changed
 - relaxations: Updated polynomial magnitude recovery to be consistent with the latest version of the arXiv paper.
 - coniclifts: Constraint violation computations (to resolve several syntax bugs which showed up in version 0.3.2).

Added
 - relaxations: An option for the user to specify skipping constrained least-squares step of solution recovery.
 - coniclifts: unittests for primal and dual SAGE cone constraint violations.


[0.3.2] - 2019-07-12
--------------------
Changed
 - coniclifts: Expression objects get value by ``.value`` instead of ``.value()``
 - coniclifts: fixed a bug in ``__contains__`` for coniclifts PrimalCondSageCone


[0.3.1] - 2019-07-09
--------------------
Changed
 - relaxations: least-squares solution recovery for polynomial problems.
 - coniclifts: conditional SAGE cones with m=2 were being compiled into overly restrictive terms,
   this is now fixed.

Added
 - relaxations: Documentation to helper functions defined in ``sageopt.relaxations`` init file.
 - relaxations: Some unittests for conditional sage polynomials.


[0.3.0] - 2019-06-30
--------------------
Removed
 - symbolic: Removed the ability to call signomials in geometric format.
 - relaxations: Removed the local_refine implementation for polynomials.

Changed
 - relaxations: Changed references to "AbK" and "logAbK" in user-facing functions to "X".

Added
 - Several functions as top-level imports in ``sageopt``.
 - symbolic: Added ``as_signomial`` function to Polynomial objects.
 - symbolic: Added ``log_domain_converter`` to ``sage_polys.py``.
 - relaxations: a function ``local_refine_polys_from_sigs``.
 - relaxations: Track the constraint functions which generate the set ``X`` in conditional SAGE
   relaxations.


[0.2.0] - 2019-05-24
--------------------
Bumping version from 0.1 to 0.2, because I've made a ton of changes to 0.1 without noting them in a changelog.
This is effectively me starting from scratch with version numbers, in preparation for a public release.

Added
 - This changelog.md file.
 - A README file.
 - License information.
