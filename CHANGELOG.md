# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
 - Added important functions to sageopt's ``__init__.py`` file.
 For example, you can now call ``sageopt.standard_sig_monomials(n)`` and
 ``sageopt.sig_constrained_primal(....)`` without following a chain of
 subpackages.
 - Removed the ability to call signomials in geometric format.
 - Removed the local_refine implementation for polynomials; replaced
   it by a generic implementation which works for polynomials and signomials
 - Added a function "local_refine_polys_from_sigs", which performs local refinement
   as though given signomial problem data actually defined polynomials. This
   is to help people who only use signomials as a modeling tool for polynomial
   optimization problems where decision variables must be nonnegative.
  - Changed references to "AbK" and "logAbK" in user-facing functions to "X".
  This is both out of consistency with the paper, and to reflect the fact that
  the set X carries more information than just a conic representation.
  - Track the constraint functions which generate the set "X" in conditional SAGE
  relaxations. By keeping track of these functions, we can check membership in X
  just be evaluating functions, rather than by solving an optimization-based
  feasibility problem. This is especially useful for polynomial problems.
  - Plan to release this next version as "0.3.0". This is done so that if bugs
  are found in 0.2.0, we can release 0.2.x which still follows the 0.2.0 API.
  The desire to have minimal support for the 0.2.0 API stems from the fact that
  I have a massive collection of experiment / simulation code that I'd rather
  not rewrite, and that I may have to run again before the paper is published.
  - Changed function names throughout.




## [0.2.0] - 2019-05-24
### Added
 - This changelog.md file.
 - A README file.
 - License information.
### Remarks
 - Bumping version from 0.1 to 0.2, because I've made a ton of changes to 0.1 without noting them in a changelog.
   This is effectively me starting from scratch with version numbers, in preparation for a public release.
