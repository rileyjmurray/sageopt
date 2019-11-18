# Planned changes to sageopt

When substantial effort is undertaken to change sageopt, or add new features, the plans and
rationales for those changes should be listed here.

## Increase compatibility with CVXPY (eventually, allow a CVXPY backend).

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

### smaller steps

Make Problem objects take two arguments instead of three;
the objective sense shouldn't be an extra argument.

Try to figure out how to do signomial arithmetic without as
many affine operations on the coefficient vector. Indexing
and stacking *will* slow things down in cvxpy.

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


## Add more dimension reduction for constrained SAGE relaxations

This note concerns signomial-constrained and polynomial-constrained
SAGE relaxations. For these problems, the Lagrangian is not taking advantage
of any sign information for its non-constant terms. For problems
where the objective function is sparse, this sign information may be
crucial. There are two clear obstacles for implementing this dimension
reduction:

1. the signs restrictions on the Lagrangian's coefficient vector can
   also affect sign restrictions on dual variables' coefficient vectors.
   So there is a cyclic dependence between dimension reduction for the Lagrangian,
   and dimension reduction for the dual variables appearing in the Lagrangian.

2. Coniclifts does not have support for tracking the signs of non-constant
   expressions.

There is also a potential drawback of implementing this kind of dimension
reduction: it would make the implementations of sig_constrained_relaxation
or sig_relaxation much harder to read. Right now the simplicity of these
functions is important, because they serve as a template if someone wanted
to implement variations of those functions. If these functions are made
substantially more complicated, then it might be good to put the current
(simple) forms of these functions in an "advanced examples" section.

###  Introduce a pre-compile function for Constraint objects.

For fancy constraints like PrimalSageCone or DualSageCone, this would
declare the auxiliary variables. The idea is that if auxiliary variable
declaration can be deferred until a Problem class sees the constraints,
more advanced presolve can be applied with all of those constraints in
mind at the same time.

For ElementwiseConstraint objects, this would recompute ``con.expr``
(and thereby undo any epigraph-substitution performed in a previous
compilation phase).

## Add verobsity settings to problem construction

It can take a long time to build bigger SAGE relaxations,
especially when an optimization-based presolve is employed.
A user might have no idea if it's worth waiting for the problem
to be built. For example, if it takes 10 minutes to construct only
a tiny fraction of the full problem, the most likely situation is
that the final SAGE relaxation can't be solved in a reasonable
time frame. If a user knew that they were only a tiny way into
problem construction, they could (rightly) terminate the program
(or otherwise interrupt execution) and try different settings.

## More powerful abstractions for Signomial and Polynomial objects

I think one hurdle for users is specifying lists of Signomial
or Polynomial objects in a standard format (``gts``, ``eqs``).
I can imagine a ``NominalConstraint`` object which is created
with operator-overloading (``==``, ``<=``). A hurdle here is
that Signomial and Polynomial objects already use ``==`` as
a test for symbolic equality.

Another hurdle to users is having to work with scalar-valued Signomial
and Polynomial objects. For more complicated models, a ``NomialMap``
object could be very useful. A ``NomialMap`` object could also be
substantially more efficient than an array of Signomial or Polynomial
objects. Operations like taking the dot-product of two Signomial
maps could easily reduce quadratic time complexity down to linear
complexity, or even faster.


