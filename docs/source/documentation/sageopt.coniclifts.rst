.. _coniclifts:

``coniclifts`` is ``sageopt``'s backend
=======================================

Sageopt does not solve SAGE relaxations on its own; it relies on third party convex optimization solvers, such as
ECOS or MOSEK. These solvers require input in very specific standard-forms. *Coniclifts*
provides abstractions that allow us to state SAGE relaxations in high-level syntax, and manage
interactions with these low-level solvers.

Overview
--------

Coniclifts employs abstractions that are similar to those in `CVXPY <cvxpy.org>`_.
For example, it is possible to construct and solve a linear program using coniclifts::

   import sageopt.coniclifts as cl
   import numpy as np
   # random problem data
   G = np.random.randn(3, 6)
   h = A @ np.random.rand(6)
   c = np.random.rand(6)
   # input to coniclift's Problem constructor
   x = cl.Variable(shape=(6,))
   constrs = [0 <= x, G @ x == h]
   objective_expression = c @ x
   prob = cl.Problem(cl.MIN, objective_expression, constrs)
   prob.solve(solver='ECOS', verbose=False)
   x_opt = x.value

In the example above, ``constrs`` consists of two coniclifts *Constraint* objects. Both of these
objects were constructed with operator overloading (``<=`` and ``==``), and the second Constraint made use of
operator-overloaded matrix multiplication (``@``). If you check the datatype of the object ``y = A @ x``, you
would find that ``y`` is a coniclifts *Expression*. Expression objects track functions of *Variable* objects.

Coniclifts has support for nonlinear convex constraints. The most important of these constraints are specified as
"set membership", rather than elementwise inequalities.
For example, we have a *PrimalSageCone* class, which is used to construct some of the SAGE
relaxations in the ``sageopt.relaxations`` package. Here is a concrete demonstration ::

   alpha = np.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1],
                     [0.5, 0],
                     [0, 0.5]])
   c = np.array([0, 3, 2, 1, -4, -2])
   # alpha, c define a signomial in the usual way
   gamma = cl.Variable(shape=(), name='gamma')
   c_expr = cl.Expression(c.copy())
   c_expr[0] -= gamma   # shift the constant term by -gamma
   constr = cl.PrimalSageCone(c_expr, alpha, X=None, name='example_constraint')
   prob = Problem(cl.MAX, gamma, [constr])
   # find largest gamma so shifted signomial is nonnegative
   status, val = prob.solve()

By solving the problem described above, we have that ``val`` is a lower bound on the signomial which takes values
``lambda x: c @ np.exp(alpha @ x)``.
There are only a few set-membership constraints currently implemented in
coniclifts; they can all be found in ``sageopt.coniclifts.constraints.set_membership``.

Coniclifts contains *operators* which represent functions applied to Expression objects.
The module ``sageopt/coniclifts/operators/affine.py`` contains all affine array-manipulation operators you would use on
a numpy ndarray `[link] <https://github.com/rileyjmurray/sageopt/blob/master/sageopt/coniclifts/operators/affine.py>`_.
This includes linear algebra operators such as ``kron`` or ``trace``, reshaping
operators such as ``hstack`` or ``tile``, and array-creation routines like ``diag`` or ``triu``.
These functions behave in *identical* ways to their numpy counterparts, because Expression objects are actually a
custom subclass of numpy's ``ndarray`` datatype.

Coniclifts also has a small selection of nonlinear operators:
``weighted_sum_exp``, ``vector2norm``, and ``relent``. It is easy to add more nonlinear operators, but these three
suffice for the internal uses currently found in sageopt. Here is a concrete example ::

   alpha = np.array([[1, 0],
                     [0, 1],
                     [1, 1],
                     [0.5, 0],
                     [0, 0.5]])
   c = np.array([3, 2, 1, 4, 2])
   x = cl.Variable(shape=(2,), name='x')
   cons = [cl.weighted_sum_exp(c, alpha @ x) <= 1]
   obj = -x[0] - 2*x[1]
   prob = Problem(cl.MIN, obj, cons)
   prob.solve()
   x_expect = np.array([-4.93083, -2.73838])
   x_actual = x.value
   print(np.allclose(x_expect, x_actual, atol=1e-4))

If you run the code above, you should find that it prints ``True``.

It is important to note that nonlinear
operators are not allowed in the objective function. So if you want to minimize a nonlinear convex function given by
``expr``, you need to create an auxiliary variable such as ``t = Variable(shape=(1,))``, add the constraint ``expr
<= t``, and set the objective to minimize ``t``. Here is an example of a constrained least-squares problem we solve
for solution recovery in dual SAGE relaxations ::

   A, b, K = con.X.A, con.X.b, con.X.K  # con is a DualSageCone instance
   log_v = np.log(con.v.value)
   A = np.asarray(A)
   x = cl.Variable(shape=(A.shape[1],))
   t = cl.Variable(shape=(1,))
   cons = [cl.vector2norm(log_v - alpha @ x[:con.n]) <= t,
           cl.PrimalProductCone(A @ x + b, K)]
   prob = cl.Problem(cl.MIN, t, cons)

The example above also alludes to a useful set-membership constraint, called ``PrimalProductCone``. Follow
`this link <https://github.com/rileyjmurray/sageopt/tree/master/sageopt/coniclifts/constraints/set_membership>`_
for source code of set-membership constraints available in coniclifts.

The Variable class
------------------

.. autoclass:: sageopt.coniclifts.Variable
    :members:

The Problem class
-----------------

Detailed documentation for the Problem class is given below. Most users will only interact with
a few aspects of Problem objects. On a first read, it should be enough just to skim the documentation
for this class. If you want to understand all the attributes of a Problem object,
you will need to read :ref:`cl_expression_system` and :ref:`cl_compilerinterface`.

.. autoclass:: sageopt.coniclifts.Problem
    :members:

.. _cl_expression_system:

The Expression system
---------------------

Coniclifts is built around a few core ideas, including ...

- transparency in the compilation process,
- ease-of-extension for experts in convex optimization,
- no dependence on a C or C++ backend,
- full compatibility with numpy.

In order to achieve full compatibility with numpy, coniclifts takes an elementwise approach to symbolic expressions.
Specifically, coniclifts begins with a few simple abstractions for scalar-valued symbolic expressions, and wraps
those abstractions in a custom subclass of numpy's ndarray. The coniclifts abstractions for scalar-valued symbolic
expressions are as follows:

- A *ScalarExpression* class represents scalar-valued affine functions of certain irreducible primatives.
  ScalarExpressions are operator-overloaded to support ``+``, ``-``, and ``*``. This allows ndarrays of
  ScalarExpressions to fall back on many functions which are implemented for numeric ndarrays.

- An abstract *ScalarAtom* class specifies the behavior of the irreducible primitives in ScalarExpressions. The
  ScalarAtom class immediately specializes into *ScalarVariables* (far and away the most important ScalarAtom), and
  another abstract class, called *NonlinearScalarAtom*. NonlinearScalarAtoms are implemented on a case-by-case basis,
  but include such things as the exponential function and the vector 2-norm.

We ask interested users to refer to the source code for additional information on ScalarExpressions and
ScalarAtoms. For most people, all you need to work with is the Expression class.


.. autoclass:: sageopt.coniclifts.Expression
    :members:


SAGE constraints
----------------

.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614


Coniclifts provides direct implementations of the primal and dual signomial SAGE cones:

 - :class:`sageopt.coniclifts.PrimalSageCone`, and
 - :class:`sageopt.coniclifts.DualSageCone`.

These classes have virtually identical constructors and public attributes. In particular, both classes' constructors
require
an argument ``X``, which can be ``None`` or a ``SigDomain``.
Ordinary SAGE constraints are obtained by setting ``X=None``.
Conditional SAGE constraints assume the set represented by ``X`` is nonempty, and it's the user's
responsibility to ensure this is the case.
The main difference in these classes' attributes is
that ``PrimalSageCone`` instances have a dict called ``age_vectors`` (which represent the certificates of nonnegativity)
and that ``DualSageCone`` instances have a dict called ``mu_vars`` (which are useful for solution recovery in SAGE
relaxations of signomial programs).

The ``PrimalSageCone`` class performs a very efficient dimension-reduction procedure by analyzing the signs of
the provided vector ``c``. The details of the reduction are described in Corollary 5 of MCW2019_.
At present, coniclifts does not provide a means to track the signs of Variable objects, and so this reduction
is limited to indices ``i`` where ``c[i]`` is constant. This feature can optionally be carried over to
``DualSageCone`` objects, if the user provides a keyword argument ``c`` to the ``DualSageCone`` constructor.

The ``PrimalSageCone`` and ``DualSageCone`` classes can perform a more extensive presolve phase to
eliminate trivial AGE cones (those which reduce to the nonnegative orthant).
By default, this presolve ability is turned off. This default can be changed by calling
``sageopt.coniclifts.presolve_trivial_age_cones(True)``.
The computational cost of this presolve is borne when the constraint is constructed, and scales linearly in the
dimension of the SAGE constraint (equal to ``constr.alpha.shape[0]``).
The cost of this presolve can be mitigated by recycling ``covers = constr.ech.expcovers`` from one call of a
constraint constructor to the next.

Primal SAGE constraints
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sageopt.coniclifts.PrimalSageCone
    :members:

Dual SAGE constraints
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sageopt.coniclifts.DualSageCone
    :members:

.. _cl_sage_options:

Compilation options
~~~~~~~~~~~~~~~~~~~

Coniclifts makes several decisions when compiling a SAGE constraint into a form
which is acceptable to a solver like MOSEK or ECOS. The following functions allow
you to control the defaults for this compilation process. The defaults can always
be overridden by providing an appropriate keyword argument to the ``PrimalSageCone``
or ``DualSageCone`` constructor. Regardless of whether or not the default values
are overridden, the settings used in a ``PrimalSageCone`` or ``DualSageCone`` object
are cached upon construction. Therefore it is safe to modify these defaults while
constructing different constraints for use in the same model.

.. autofunction:: sageopt.coniclifts.presolve_trivial_age_cones

.. autofunction:: sageopt.coniclifts.heuristic_reduce_cond_age_cones

.. autofunction:: sageopt.coniclifts.age_cone_reduction_solver

.. autofunction:: sageopt.coniclifts.sum_age_force_equality

.. autofunction:: sageopt.coniclifts.compact_sage_duals

.. autofunction:: sageopt.coniclifts.kernel_basis_age_witnesses

.. _cl_compilerinterface:

The compiler interface
----------------------

Up until now we have only described coniclifts as a tool for creating optimization problems. However, coniclifts'
more fundamental use is to exploit the following fact: for every convex set :math:`X \subset R^n`, there exists a
matrix :math:`A \in R^{k \times m}` , a vector :math:`b \in R^k`, and a convex cone :math:`K \subset R^k` so that
:math:`X = \{ (x_1,\ldots,x_n) \,:\, A x + b \in K, x \in R^m \}`. Coniclifts compiles all optimization problems into
this standard form, where :math:`K` is a product of elementary convex cones

#. The zero cone.
#. The nonnegative orthant.
#. The exponential cone.
#. The second-order cone.
#. The vectorized positive semidefinite cone.


Crucially, coniclifts provides a means to map back and forth
between models specified in high-level syntax, and models which exist in a flattened conic form using only primitives
above.

The most important function in coniclifts' compilation process is given below.
The final return argument mentions "ScalarVariable" objects, which users of coniclifts need not interact with directly.

.. autofunction:: sageopt.coniclifts.compilers.compile_constrained_system
