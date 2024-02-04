# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.6.0] - unreleased
## Added
 - symbolic: you can now call a ``Polynomial`` object by providing an array of ``Signomial`` objects.
 - symbolic: the function ``Signomial.shift_coordates``.
 - relaxations: polynomial SAGE constraints now use ``covers`` in a way that
   reflects the signs of a signomial representative coefficient vector.    
 - coniclifts: CVXPY can be called as a "meta solver" (used to access power cones and solvers other than
   ECOS/MOSEK).
 - coniclifts: added a hotfix to the ``__init__`` function of ``PowCone3D`` objects in CVXPY. This change
   is needed due to a change in CVXPY in late 2023. This bugfix will be ported to cvxpy in a timely way
   and so this change can be reverted in the next sageopt release.
## Changed
 - coniclifts: how ``disjoint_dot`` is computed in ``coniclifts.base``.
 - interop: implementation of the ``__array_priority__`` hotfix for scipy sparse matrices.
 - interop: GPKit models weren't converted deterministically.
 - interop: CVXPY's ``*`` (star) operator is no longer overridden to use elementwise multiplication.
 - relaxations: ``coniclifts.clear_variable_indices()`` is no longer called by default
   at the end of generating SAGE relaxations. The old behavior can be restored by setting
   ``sage_polys.AUTO_CLEAR_INDICES = True`` and ``sage_sigs.AUTO_CLEAR_INDICES = True``.
## Removed
 - symbolic: CVXPY Expression objects can no longer be used as coefficients or exponent vectors in
   Signomial or Polynomial objects.
 - interop: in light of the change above, we no longer needed a CVXPY interop file.

# [0.5.3] - 2020-09-07
## Changed
 - coniclifts: Fix regression introduced in previous version (optimization based dimension reduction was
   inadvertently disabled).
 - coniclifts: Remove an unstated assumption in the ``relent.py`` precompiled atom (and ``DualSageCone.compile``)
   that certain inputs had to be linear (rather than affine) expressions of decision variables.
 - coniclifts: Minor change to how DualSageCones compute violations.

# [0.5.2] - 2020-07-05
## Added
 - coniclifts: the mosek interface now automatically dualizes certain problems.
 - coniclifts: ``prob.solve`` accepts an argument to override or force dualization (when using MOSEK).
## Changed
 - coniclifts: heavily refactor precompiled affine atoms.
 - relaxations: dual SAGE relaxations no longer introduce high-level slack variables
   by default.

# [0.5.1] - 2020-06-21
## Added
 - coniclifts: "abs" and "pos" NonlinearScalarAtoms.
 - coniclifts: a field "age_witnesses" to PrimalSageCone objects.
 - coniclifts: compile_objective.
 - coniclifts: an option to have ``nu`` (an AGE witness) be defined by substituting
   ``nu = M_k @ u_k`` where ``M_k`` is a matrix whose columns form a basis for the nullspace
   of an appropriate matrix. Eliminates the need for equality constraints in ordinary AGE cones.
 - coniclifts: PrimalSageCone objects have a ``sigma_x`` method, for evaluating support function.
   This method is now used when computing constraint violations.
 - symbolic: SigDomain objects have a ``suppfunc`` method.
## Changed
 - symbolic: signomials will now broadcast to meet the size of an ndarray,
   if the ndarray cannot be coerced into a scalar.
## Removed
 - coniclifts: compile_linear_expression.


# [0.5.0] - 2020-03-25
## Added
 - symbolic: ``Signomial.from_dict`` and ``Polynomial.from_dict`` convenience functions for construction.
 - relaxations: keyword argument checking in ``sig_constrained_relaxation`` and ``poly_constrained_relaxation``.
 - coniclifts: an option to reduce epigraph variable usage in dual SAGE relaxations.
 - coniclifts: mixed-integer support for MOSEK.
 - coniclifts: ``PrimalSageCone.settings`` and ``DualSageCone.settings``: dictionary-valued properties which record
   compilation settings (e.g. pre-solve behavior)
 - coniclifts: Properly expose all MOSEK solver options.
 - interop: Proper tests for GPKit integration.
 - interop: CVXPY integration.
## Changed
 - symbolic: Signomials are now represented primarily by arrays, rather than a dictionary.
   The dictionary is now only constructed when necessary, and is not used in any signomial
   arithmetic operations.
 - symbolic: Signomial and Polynomial constructors: both parameters ``alpha`` and ``c`` are *required*.
## Removed
 - coniclifts: private compilation constants (e.g. ``_REDUCTION_SOLVER_``) in ``sage_cones.py``
 - coniclifts: private solver parameters (e.g. ``_DEACTIVATE_SCALING``) in ``mosek.py``


# [0.4.2] - 2019-11-18
## Added
 - A ``metadata`` field to Signomial and Polynomial objects.
 - ``sageopt.interop``, for interacting with other python projects. Right now, there
   is only support for GPKit. The GPKit functions are minimal, and untested. They
   will be properly fleshed out when version 0.5 is released.
## Changed
 - ``Signomial.partial`` and ``Polynomial.partial`` to correctly work with nonconstant coefficients.
 - Essentially all "attributes" of Signomial or Polynomial objects have become "properties". The
   most important change is that ``self.alpha`` and ``self.c`` are now only created when necessary;
   the internal ``alpha_c`` dict still defines the Signomial or Polynomial as a mathematical object,
   but this has also been turned into a hidden attribute ``self._alpha_c``.

# [0.4.1] - 2019-10-13
## Changed
 - Fixed a bug in the function ``clcons_from_standard_gprep``.
 - ``Variable.value`` is now a settable property.
 - Fixed a bug in SigDomain and PolyDomain.
## Removed
 - ``Variable.set_scalar_variables``.
 - The ``name`` field of the ScalarVariable class.
 - Some unused functions in ``coniclifts.base.ScalarAtom``.
## Added
 - An ``index`` field to the ScalarVariable class.
 - A ``_DEACTIVATE_SCALING_`` static attribute in the ``Mosek`` class.

# [0.4.0] - 2019-10-06
## Added
 - SigDomain and PolyDomain classes.
 - sage_cones.py, which handles the ordinary and conditional SAGE cases.
 - Support for automatic elimination of trivial AGE cones from SAGE relaxations. This can be disabled
   by calling ``sageopt.coniclifts.presolve_trivial_age_cones(False)``.
 - An explicit requirement that ``Constraint.variables()`` returns both all variables in its scope,
   and that all returned Variables be "proper".
 - For PrimalSageCone: added the option to allow AGE vectors to sum <= c, or == c. The default is <= c.
   The default setting can be changed by calling ``sageopt.coniclifts.sum_age_force_equality(True)``.
## Changed
 - ``conditional_sage_data`` is ``infer_domain``. These functions have an additional argument for checking
   feasibility of the inferred system (defaults to True).
 - The signature of ``sig_relaxation`` and ``sig_constrained_relaxation``. Hierarchy parameters are now specified
   by keyword arguments, and the argument ``X`` comes before the argument ``form``.
 - Conditional SAGE constraints now assume that the "conditioning" is feasible.
 - The ``var_name_to_locs`` dict from ``coniclifts.compilers.compile_constrained_system``.
   It now has a slightly different definition, and a different name.
 - Fields in the Problem class. ``Problem.user_cons`` is now ``Problem.constraints``.
   ``Problem.user_obj`` is now ``Problem.objective_expr``.
   ``Problem.user_variable_map`` is now ``Problem.variable_map``.
 - How ElementwiseConstraint objects report their variables. They now always include user-defined variables,
   and if ``con.variables()`` is called after epigraph substitution, then the list will also contain the epigraph
   variables.
 - Fixed a bug in ``conditional_sage_data`` (now, ``infer_domain``). Equality constraints were being
   incorrectly compiled. The bug only meant that SAGE relaxations solved in the past were unnecessarily weak.
## Removed
 - ``log_domain_converter``. It wasn't being used.
 - conditional_sage_cone.py, sage_cone.py.
 - sig_primal, sig_dual, poly_primal, poly_dual (and the four constrained variations thereof)
   as top-level imports within sageopt. These functions are still accessible from sageopt.relaxations.


# [0.3.4] - 2019-09-09
## Added
 - sig_relaxation and sig_constrained_relaxation. These are wrappers around sig_primal/sig_dual
   and sig_constrained_primal/sig_constrained_dual. I introduced these because now users can specify the
   form as a keyword argument. This keeps the number of user-facing functions lower, and
   the docstrings for these functions can be heavy on LaTeX for rendering in web documentation.
 - Variants of the above for polynomial problems.
## Changed
 - Major changes to how coniclifts handles the compilation process. Refer to commit 7cb07866e55c6618ce17d090d52281512ea2351f.


# [0.3.3] - 2019-08-10
## Added
 - unittests for primal and dual SAGE cone constraint violations.
 - An option for the user to specify skipping constrained least-squares step of solution recovery.
## Changed
 - Constraint violation computations (to resolve several syntax bugs which showed up in version 0.3.2).
 - Updated polynomial magnitude recovery to be consistent with the latest version of the arXiv paper.


# [0.3.2] - 2019-07-12
## Changed
 - coniclifts Expression objects get value by ``.value`` instead of ``.value()``
 - fixed a bug in ``__contains__`` for coniclifts PrimalCondSageCone


# [0.3.1] - 2019-07-09
## Added
 - Documentation to helper functions defined in ``sageopt.relaxations`` init file.
 - Some unittests for conditional sage polynomials.
## Changed
 - Conditional SAGE cone compilation behavior. It used to be that conditional
   SAGE cones were replaced by the nonnegative orthant if the parameter ``m <= 2`` .
   (The <= 2 was a hold-over from ordinary SAGE cones, where any 2-dimensional SAGE cone
   is equal to R^2_+). The behavior is now corrected, so that 2-dimensional conditional
   SAGE cones compile to mathematically correct forms.
 - Solution recovery for SAGE polynomial relaxations. The MCW2019 paper didnt use
   constrained least-squares, because there was a separate need to handle when some
   entries of the moment vector were zero. The new implementation now solves a
   constrained least-squares problem when all entries of the moment vector are nonzero.


# [0.3.0] - 2019-06-30
## Added
 - Important functions to sageopt's ``__init__.py`` file.
   For example, you can now call ``sageopt.standard_sig_monomials(n)`` and
   ``sageopt.sig_constrained_primal(....)`` without following a chain of
   subpackages.
 - Added a function ``local_refine_polys_from_sigs``, which performs local refinement
   as though given signomial problem data actually defined polynomials. This
   is to help people who only use signomials as a modeling tool for polynomial
   optimization problems where decision variables must be nonnegative.
 - Track the constraint functions which generate the set ``X`` in conditional SAGE
   relaxations. By keeping track of these functions, we can check membership in ``X``
   just be evaluating functions, rather than by solving an optimization-based
   feasibility problem. This is especially useful for polynomial problems.
 - Added a ``as_signomial`` function to Polynomial objects. There isn't a specific
   use-case for this in sageopt's codebase, but it makes the Polynomial API closer
   to the Signomial API, and that's desirable in its own right.
 - Added a ``log_domain_converter`` to ``sage_polys.py``. The purpose of this new function
   is to allow arguments to be passed in log-domain, but evaluated as though they
   were in the original function's domain. This mimics the functionality of a user
   calling ``f_sig = f.as_signomial()`` and subsequently evaluating ``f_sig(x)``,
   however ``log_domain_converter`` does not require inputs to be Polynomial objects.
## Changed
 - Removed the local_refine implementation for polynomials; replaced
   it by a generic implementation which works for polynomials and signomials
 - Changed references to "X" and "logAbK" in user-facing functions to "X".
   This is both out of consistency with the paper, and to reflect the fact that
   the set X carries more information than just a conic representation.
## Removed
 - Removed the ability to call signomials in geometric format.
## Remarks
 - Plan to release this version as "0.3.0". This is done so that if bugs
   are found in 0.2.0, we can release 0.2.x which still follows the 0.2.0 API.
   The desire to have minimal support for the 0.2.0 API stems from the fact that
   I have a massive collection of experiment / simulation code that I'd rather
   not rewrite, and that I may have to run again before the paper is published.


## [0.2.0] - 2019-05-24
### Added
 - This changelog.md file.
 - A README file.
 - License information.
### Remarks
 - Bumping version from 0.1 to 0.2, because I've made a ton of changes to 0.1 without noting them in a changelog.
   This is effectively me starting from scratch with version numbers, in preparation for a public release.
