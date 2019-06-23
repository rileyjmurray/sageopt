clifts

For any convex set ``X`` in R^n, there exists a matrix ``A``, a vector ``b``, a cone ``K``, and a projector ``P`` so that ``X = { P * z : A * z + b \in K }``. The projection operator can be taken to be something that grabs the first ``n`` coordinates of ``z``. The purpose of this package is allow the user to define ``X`` in a convenient way (such as how one would define an optimization model in cvxpy), then generate the data ``A,b,K`` defining the conic lift of ``X``.

The original design goals were as follows:

* No dependence on C or C++ backend.
* Full compatibility with numpy (i.e. affine numpy operations such as kron, diag, triu, stack, split, ... can be used natively on symbolic expressions used to define convex sets).
* Experts in optimization should be able to easily extend it to add additional functionality if needed.
* Ammenable to unittests without invoking solvers.

The current form of coniclifts has significantly expanded. It now features "Problem" and "Solver" classes, much like cvxpy. Coniclifts is currently the exclusive backend rewriting system for sageopt. Although it is good at what it does, its functionality is still limited.

* Coniclifts does not propogate dual variables through the rewriting system.
* Coniclifts is not well-suited to setting up and solving auxilliary optimization problems as part of constructing a larger, master optimization problem.

This second feature is actually quite important for dimension reduction and preprocessing in SAGE relaxations.
