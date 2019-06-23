# Sageopt is for signomial and polynomial optimization

[![Build Status](https://travis-ci.org/rileyjmurray/sageopt.svg?branch=master)](https://travis-ci.org/rileyjmurray/sageopt)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sageopt.svg)

Sageopt provides functionality for constructing, solving, and analyzing convex relaxations for
signomial and polynomial optimization problems. It also provides functionality for recovering feasible
solutions from these convex relaxations. This means that sageopt can be used as a standalone tool to find provably
optimal solutions to hard optimization problems, or as part of a broader effort to find locally-optimal
solutions to especially difficult problems (with bounds on possible optimality gaps).

These underlying convex relaxations are built upon the idea of "SAGE certificates" for signomial and
polynomial nonnegativity. Refer to the paper "Signomial and Polynomial Optimization by Relative Entropy
and Partial Dualization" for a mathematical description of the functionality implemented by this python package.

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

## Mathematical background

SAGE certificates were originally developed for signomials ([Chandrasekaran and Shah, 2016](https://arxiv.org/abs/1409.7640)).
Around the same time that SAGE certificates were first developed, there were similar developments for polynomials, 
under the name "SONC certificates." Work by Murray, Chandrasekaran, and Wierman ([MCW 2018](https://arxiv.org/abs/1810.01614))
described the first efficient way to use these SONC certificates in optimization; they referred to their efficient proof
 system as the method of "SAGE polynomials."

 The appendix of the 2018 paper by MCW describes a python package called
 "sigpy", which implements SAGE relaxations for both signomial and polynomial optimization problems. Sageopt 
 supercedes sigpy by implementing a significant generalization of the original SAGE certificates for both signomials 
 and polynomials. The formal name for these  generalizations are "conditional SAGE signomials" and "conditional SAGE
  polynomials". This concept is introduced in a 2019 paper by Murray, Chandrasekaran, and Wierman, titled "Signomial 
  and Polynomial Optimization via Relative Entropy and Partial 
 Dualization." Because the generalization follows so transparently from the original idea of SAGE certificates, we say simply say "SAGE
 certificates" in reference to the most general idea.


