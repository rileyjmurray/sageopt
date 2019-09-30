# Planned changes to sageopt

When substantial effort is undertaken to change sageopt, or add new features, the plans and
rationales for those changes should be listed here.


## Add tests for constraint violations (coniclifts)

This is a low-priority task.

The ability to compute constraint violations was added pretty late in the coniclifts development process.
Linear equations and inequalities, and SAGE cones have tests. Should add tests for other
SetMembership constraints, and for some nonlinear atoms (such as vector2norm, or exp).

## Populate the design_notes folder.

This is a high-priority task. It can be done incrementally.

Files in the design_notes folder should explain the overall architecture of sageopt and its
various submodules (e.g. coniclifts). These files should serve as both a guide to those who
are trying to understand how sageopt works for the first time, and as a reference for people
who are already deeply familiar with certain aspects of sageopt. These documents should keep
track of "lessons learned" throughout the life of the sageopt project (e.g. "We used to do X,
which was nice for Y reasons, but created problems for Z, so now we do W.")


## Create a flexible backend, that can use either coniclifts, or cvxpy.

This is a high-priority task. It will require large, coordinated changes to sageopt.

An earlier version of sageopt (called "sigpy") used cvxpy as its backend for constructing SAGE
relaxations. We moved to a custom backend for a few reasons:
1. CVXPY had slow performance for compile-time.
2. CVXPY lacked functionality to track user-defined variables through to the end of the
   compilation process. Thus although low-level solver data could be accessed with cvxpy,
   there was no way to understand how this data mapped to the original problem variables.
3. CVXPY has a C++ backend and significant solver dependencies, which often create problems
   for users trying to install or run cvxpy.
4. I had an idea for a different architecture for a rewriting system of convex programs, and
   I wanted to test out that idea. I implemented that architecture in "coniclifts", and once
   coniclifts resolved the three problems above with cvxpy, I moved to a coniclifts backend.

All that said -- there are several features of cvxpy which are relevant to the goals of sageopt,
but that would be a terrible mess to implement with coniclifts. The list of such things is
likely going to grow with time. My time (and the time of others in the optimization community)
is much better spent helping cvxpy, rather than continuing to add features to coniclifts.

My "vision" for the rewriting system behind sageopt is roughly as follows.
1. Have the ability to run core functionality of sageopt with only coniclifts.
2. Have a submodule "cvxtensions" which adds functionality to cvxpy (by properly subclassing
   existing parts of cvxpy) that is important for sageopt.
3. Have the "relaxations" submodule contain a setting which allows the user to switch between
   coniclifts and cvxpy as desired. Possibly default to cvxpy if cvxpy is installed, but
   whether or not we do that depends on if cvxpy's compilation process is slower than coniclifts.

The third part in that vision is the fuzziest bit. I don't want to make the relaxations submodule
a mess of code that is hard to read and hard to maintain. In addition, it will be important to
make sure that the coniclifts API is aligned with the cvxpy API as best as possible.


## Add support for more solvers (coniclifts)

This is a medium-priority task. It can be done with virtually no changes to existing files. 

Once we have a cvxpy backend, we will have access to essentially all exponential-cone solvers
that might be useful for SAGE relaxations. Until then, we need to add more solver interfaces
to coniclifts. It should be easy to add support for both SCS and SuperSCS. 


## Add support for SDPs in MOSEK's coniclifts interface.

This is a low-priority task.

Coniclifts currently allows users to create linear matrix inequality constraints, but
existing solver interfaces do not allow these constraints. (So right now, a user can construct
an SDP, but cannot solve an SDP.)


## Add more sophisticated dimension reduction for SAGE cones

Suppose we want to represent a constraint "c is X-SAGE with respect to exponents alpha".
Often, there are many indices "k" where the k^th X-AGE cone w.r.t. alpha is simply the
nonnegative orthant.

If X is a cone, then it is easy to identify these indices:

find a nonzero vector x \in X so that (alpha - alpha[k,:]) @ x < 0.

When X is not a cone, we can check that

    min{ t : x in X, (alpha - alpha[k,:]) @ x <= t } = -\infty

Or, for numeric purposes, it should suffice to check that the value
of the above optimization problem is <= -1000 (since exp(-1000)
is zero in 64 bit arithmetic).

In order to add this dimension-reduction to sageopt, it will be necessary to
create "constraint factories", which perform this dimension reduction once,
and allow it to be re-used across multiple desired constraints.

*UPDATE* There is now an \_EXPENSIVE\_REDUCTION\_ flag in both ordinary_sage_cone.py and
conditional_sage_cone.py which has ExpCoverHelper objects solve these optimization
problems. Here are some useful extensions:

1. Enable performing the reductions in parallel, with Dask.
2. Recycle the expcovers across primal and dual forms of the same
SAGE relaxation.
3. Allow some logging during this reduction phase, so users have a sense
for how long it's going to take.


## Remove separate ordinary vs conditional SAGE implementations

Find a way to unify the compilation process in an efficient way.
Maybe figure out how to do this alongside CVXPY integration.
