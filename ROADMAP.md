# Planned changes to sageopt

When substantial effort is undertaken to change sageopt, or add new features, the plans and
rationales for those changes should be listed here.

## 1. Populate the design_notes folder.

Files in the design_notes folder should explain the overall architecture of sageopt and its
various submodules (e.g. coniclifts). These files should serve as both a guide to those who
are trying to understand how sageopt works for the first time, and as a reference for people
who are already deeply familiar with certain aspects of sageopt. These documents should keep
track of "lessons learned" throughout the life of the sageopt project (e.g. "We used to do X,
which was nice for Y reasons, but created problems for Z, so now we do W.")

## 2. Create a flexible backend, that can use either coniclifts, or cvxpy.

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
a mess of code that is hard to read and hard to maintain. In addition, although the coniclifts API
closely matches the cvxpy API in many ways, there are some imporant differences (for example,
accessing the value of a Variable ``x`` with ``x.value`` versus ``x.value()``). These differences
will have to be resolved.
