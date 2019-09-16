``sageopt.coniclifts`` is a basic modeling language.
====================================================

Sageopt does not solve SAGE relaxations on its own; it relies on third party convex optimization solvers, such as
ECOS or MOSEK. These solvers require input in very specific standard-forms. *Coniclifts*
provides abstractions that allow us to state SAGE relaxations in high-level syntax, and manage
interactions with these low-level solvers. These abstractions are similar to those in `CVXPY <cvxpy
.org>`_. For example, it is possible to construct and solve a linear program using coniclifts::

   import sageopt.coniclifts as cl
   import numpy as np
   # random problem data
   G = np.random.randn(3, 6)
   h = A @ np.random.rand(6)
   c = np.random.rand(6)
   # input to coniclift's Problem constructor
   x = cl.Variable(shape=(6,))
   constraints = [0 <= x, G @ x == h]
   objective_expression = c @ x
   prob = cl.Problem(cl.MIN, objective_expression, constraints)
   prob.solve(solver='ECOS', verbose=False)
   x_opt = x.value

In the example above, ``constraints`` is a list consisting of two coniclifts Constraint objects. Both of these
objects were constructed with operator overloading (``<=`` and ``==``), and the second Constraint made use of
operator-overloaded matrix multiplication (``@``). If you check the datatype of the object ``y = A @ x``, you
would find that ``y`` is a coniclifts Expression object. Expression objects symbolically track functions of Variables,
in an elementwise fashion.

Coniclifts has support for nonlinear convex constraints. The most important of these constraints are specified as
"set membership", rather than elementwise inequalities.
For example, we have a ``PrimalSageCone`` class, which is used to construct some of the SAGE
relaxations in the ``sageopt.relaxations`` package. Here is a concrete example ::

   alpha = np.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1],
                     [0.5, 0],
                     [0, 0.5]])
   c = np.array([0, 3, 2, 1, -4, -2])
   # alpha, c define a signomial in the usual way
   gamma = cl.Variable(shape=(), name='gamma')
   c_expr = c.copy()
   c_expr[0] -= gamma   # shift the constant term by -gamma
   constr = cl.PrimalSageCone(c_expr, alpha, name='example_constraint')
   prob = Problem(cl.MAX, gamma, [constr])
   # find largest gamma so shifted signomial is nonnegative
   status, val = prob.solve()

By solving the problem described above, we have that ``val`` is a lower bound on the signomial which takes values
``lambda x: c @ np.exp(alpha @ x)``.
There are only a few set-membership constraints currently implemented in
coniclifts; they can all be found in ``sageopt.coniclifts.constraints.set_membership``.

Coniclifts contains "operators" which symbolically represent functions applied to Expression objects.
The module ``sageopt.coniclifts.operators.affine`` contains all affine array-manipulation operators you would use on
a numpy ndarray. This includes linear algebra operators such as ``kron`` or ``trace``, reshaping
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