# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [unreleased]
## Addded
 - Support for automatic elimination of trivial AGE cones from SAGE relaxations. This can be disabled
 by setting the variable \_EXPENSIVE\_REDUCTION\_=False for both sage_cone.py and conditional_sage_cone.py.
 - An argument to conditional\_sage\_data functions, so that they now can check if the system
 defined by the inferred constraints is feasible. Adding this allowed me to make it an assumption
 in conditional_sage_cone.py that the conic system is feasible, which resolved a standing issue
 in computing constraint violations for conditional SAGE cones.
## Changed
 - The "var_name_to_locs" dict from coniclifts.compilers.compile_constrained_system. If a ScalarVariable
 does not participate in the conic system but its parent Variable object does, then that ScalarVariable's
 selector index in var_name_to_locs\[parent_variable.name\] will **now** be -1. In order to load values
 into user-defined Variable objects, the vector "x" for the vectorized conic system **must now be** augmented
 to "x0 = np.hstack(\[x,0\])". Augmenting in this way means that ScalarVariables are set to the value 0
 by default, but we only do this if its value does not affect the feasibility of the constraints.
 - Fixed a bug in condiational_sage_data. Equality constraints were not being correctly compiled.
 The bug only meant that SAGE relaxations solved in the past were weaker than they should have been.

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
 - coniclifts Expression objects get value by ".value" instead of ".value()"
 - fixed a bug in "\_\_contains\_\_" for coniclifts PrimalCondSageCone


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
 - Changed references to "AbK" and "logAbK" in user-facing functions to "X".
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
