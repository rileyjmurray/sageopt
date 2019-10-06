
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
and Partial Dualization <https://arxiv.org/abs/1907.00814>`_ provides a mathematical description of the functionality
implemented by this python package. That paper however is a bit long, and
so we hope that the "Examples" and "Documentation" links in the sidebar are
at least enough to get people rolling.


.. toctree::
   :maxdepth: 3

   Installation <install>

   Examples <examples/examples>

   Documentation <documentation/sageopt>

   File a Bug Report <https://github.com/rileyjmurray/sageopt/issues>

   Source Code <https://github.com/rileyjmurray/sageopt>



Some mathematical background
----------------------------

SAGE certificates were originally developed for signomials
(`Chandrasekaran and Shah, 2016 <https://arxiv.org/abs/1409.7640>`_).

Subsequent work by Murray, Chandrasekaran, and Wierman (`MCW2018 <https://arxiv.org/abs/1810.01614>`_) described how
these methods could be adapted to polynomial optimization, and they referred to their proof system as "SAGE
polynomials".  These ideas are especially well-suited to sparse polynomials or polynomials of high degree. The bounds
returned are always at least as strong as those computed by `SDSOS <https://arxiv.org/abs/1706.02586>`_, and can be
stronger. The expressive power of SAGE polyomials is equivalent to that of SONC polynomials, however there is no
known method to efficiently optimize over the set of SONC polynomials without using the SAGE proof system. For
example, the `2018 algorithm <https://arxiv.org/abs/1808.08431>`_ for computing SONC certificates is efficient, but
is also known to have simple cases where it fails to find SONC certificates that do in fact exist.

The appendix of the 2018 paper by MCW describes a python package called "sigpy", which implements SAGE relaxations
for both signomial and polynomial optimization problems. Sageopt supercedes sigpy by implementing a significant
generalization of the original SAGE certificates for both signomials and polynomials. The formal name for these
generalizations are *conditional SAGE signomials* and *conditional SAGE polynomials*. This concept is introduced in a
2019 paper by Murray, Chandrasekaran, and Wierman, titled `Signomial and Polynomial Optimization via Relative Entropy
and Partial Dualization <https://arxiv.org/abs/1907.00814>`_.
Because the generalization follows so transparently from the original idea of SAGE certificates, we say simply say
"SAGE certificates" in reference to the most general idea.