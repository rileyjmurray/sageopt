Installation
============

Sageopt requires Python version 3.5 or higher;
we also require the following packages:

1. SciPy, version >= 1.1.
2. Numpy, version >= 1.14.
3. ECOS, version >= 2.0.

It is highly recommended that you also install `MOSEK <https://www.mosek.com/>`_ (version >= 9).
MOSEK is a commercial optimization solver, and currently the only solver that is able to handle
the more interesting convex relaxations needed for optimization with SAGE certificates. If you
are in academia you can request a free `academic license <https://www.mosek.com/products/academic-licenses/>`_ for
MOSEK.


Pip users
---------

Run ``pip install sageopt``.
As an optional second step, install nose (``pip install nose``) and then run
``nosetests sageopt``.

Conda users
-----------

If you use Anaconda for Python development, do the following:

1. activate your anaconda environment,
2. run ``conda install pip``,
3. run ``pip install sageopt``.

It is important that pip be installed inside your conda environment, or sageopt
might not be detected by Anaconda Navigator (among other environment management tools).
As an optional final step, install nose (``conda install nose``) and then run
``nosetests sageopt``.

Installation from source
------------------------

Do the following:

0. Download this repository. If needed, change your directory so that you are in the same directory as
   sageopt's ``setup.py`` file.
1. Activate the Python virtual environment of your choice.
2. Run ``pip install -e .`` to install an editable version of sageopt to your current environment.
3. Run ``python -c "import sageopt; print(sageopt.__version__)"`` to verify that sageopt installed correctly.
4. Run ``pip install nose``  (or ``conda install nose``) in preparation for running unittests.
5. Run ``nosetests sageopt/tests``.