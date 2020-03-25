
Sageopt enables proof-based signomial and polynomial optimization
=================================================================

.. image:: https://coveralls.io/repos/github/rileyjmurray/sageopt/badge.svg?branch=master
   :target: https://coveralls.io/github/rileyjmurray/sageopt?branch=master

.. image:: https://travis-ci.org/rileyjmurray/sageopt.svg?branch=master
   :target: https://travis-ci.org/rileyjmurray/sageopt

.. image:: https://img.shields.io/pypi/wheel/sageopt.svg


Sageopt provides functionality for constructing, solving, and analyzing convex relaxations for
signomial and polynomial optimization problems. It also provides functionality for recovering feasible
solutions from these convex relaxations.

You can use sageopt as a standalone tool to find provably optimal solutions to hard optimization problems.
You can also use sageopt as part of a broader effort to find locally-optimal solutions to especially difficult problems
(with bounds on possible optimality gaps).

These underlying convex relaxations are built upon the idea of "SAGE certificates" for signomial and
polynomial nonnegativity. The paper `Signomial and Polynomial Optimization via Relative Entropy
and Partial Dualization <https://arxiv.org/abs/1907.00814>`_ describes the mathematics of the functionality
implemented by this python package. That paper however is a bit long, and
so we hope that the "Examples" and "Documentation" links in the sidebar are enough to get you rolling.

The full map for this site is given below.

.. toctree::
   :maxdepth: 3

   Installation <install>

   Examples <examples/examples>

   Documentation <documentation/sageopt>

   File a Bug Report <https://github.com/rileyjmurray/sageopt/issues>

   Release History <releasehistory>

   Source Code <https://github.com/rileyjmurray/sageopt>

   Background <background>
