This file contains small tasks to improve sageopt. I write things
here when they occur to me, but I don't have time to implement
them at that particular moment.

## Remove ScalarVariable names

It's probably expensive to assign a string-based subscripted
name for each ScalarVariable. The only place where I can see
this mattering, is if a Variable object is not proper, and so
it checks the main part of it's leading ScalarVariable's name
field. However this could also be accomplished by just checking
the name of the leading ScalarVariable's ``parent`` object
(since the ``parent`` object of a ScalarVariable should always
be proper). If I want to identify a ScalarVariable in a readable
way, then I can just add an "index" field to the ScalarVariable
class, which represents that ScalarVariable's position in its
proper Variable parent.

## Change presolve behavior for trivial AGE cones

Only presolve-away the trivial AGE cones if the user asks for it.
(Since it *really* slows down problem construction.) Update unittests
so that code path is still tested with ECOS. Make sure ECOS can still
solve the resulting problems (since I expect them to have worse
conditioning). Update rst files and web documentation.

## Add unittests for reformulators.py

These functions are only called in the MOSEK code path. Since
TravisCI builds don't have MOSEK, these are not being tested
in CI builds.
