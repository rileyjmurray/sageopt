# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [Unreleased]


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
