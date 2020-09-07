# Sageopt is for signomial and polynomial optimization

[![Coverage Status](https://coveralls.io/repos/github/rileyjmurray/sageopt/badge.svg?branch=master)](https://coveralls.io/github/rileyjmurray/sageopt?branch=master)
[![Build Status](https://travis-ci.org/rileyjmurray/sageopt.svg?branch=master)](https://travis-ci.com/rileyjmurray/sageopt)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sageopt.svg)
[![DOI](https://zenodo.org/badge/182453629.svg)](https://zenodo.org/badge/latestdoi/182453629)


Sageopt provides functionality for constructing, solving, and analyzing convex relaxations for
signomial and polynomial optimization problems. It also provides functionality for recovering feasible
solutions from these convex relaxations.

You can use sageopt as a standalone tool to find provably optimal solutions to hard optimization problems.
You can also use sageopt as part of a broader effort to find locally-optimal solutions to especially difficult problems
(with bounds on possible optimality gaps).

These underlying convex relaxations are built upon the idea of "SAGE certificates" for signomial and
polynomial nonnegativity. Refer to the paper [Signomial and Polynomial Optimization via Relative Entropy
and Partial Dualization](https://arxiv.org/abs/1907.00814) for a mathematical description of the functionality
implemented by this python package.

This readme file contains a minimal amount of information on sageopt. Users of this software
are encouraged to visit [the main sageopt website](https://rileyjmurray.github.io/sageopt/).

## Dependencies

Sageopt requires Python version 3.5 or higher. There is no way around this: we make heavy use of the ``@``
operator for matrix multiplication, and this operator was only introduced in Python 3.5.
We also require the following packages
1. SciPy, version >= 1.1.
2. Numpy, version >= 1.14.
3. ECOS, version >= 2.0.

It is highly recommended that you also install [MOSEK](https://www.mosek.com/) (version >= 9).
MOSEK is a commerical optimization solver, and currently the only solver that is able to handle
the more intersting convex relaxations needed for optimization with SAGE certificates. If you
are in academia (that includes undergraduates!) you can request a free
[academic license](https://www.mosek.com/products/academic-licenses/) for MOSEK.

## To install

Run ``pip install sageopt``.

If you use Anaconda for Python development, please (1) activate your anaconda environment, (2) run ``conda install pip``, and (3) run
``pip install sageopt``. It is important that pip be installed inside your conda environment, or sageopt
might not be detected by Anaconda Navigator (among other environment management tools).

To install sageopt from source, do the following:

0. Download this repository. If needed, change your directory so that you are in the same directory as
   sageopt's ``setup.py`` file.
1. Activate the Python virtual environment of your choice.
2. Run ``python setup.py install`` to install sageopt to your current Python environment.
3. Run ``python -c "import sageopt; print(sageopt.__version__)"`` to verify that sageopt installed correctly.
4. Run ``pip install nose``  (or ``conda install nose``) in preparation for running unittests.
5. Run ``nosetests sageopt/tests``.

