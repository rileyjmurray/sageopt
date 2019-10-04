.. toctree::
   :maxdepth: 1

.. _allexamples:

Examples
========

The examples shown here appear in `the 2019 article <https://arxiv.org/abs/1907.00814>`_ by Murray, Chandrasekaran,
and Wierman, titled *Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization*. That paper
introduced conditional SAGE certificates.

Optimization Examples
---------------------


Example 1 (signomial)
~~~~~~~~~~~~~~~~~~~~~

We want to solve a constraint signomial program in three variables.

.. math::

    \begin{align*}
     \min_{x \in \mathbb{R}^3} &~ f(x) \doteq 0.5 \exp(x_1 - x_2) -\exp x_1  - 5 \exp(-x_2)  \\
                            \text{s.t.} &~ g_1(x) \doteq 100 -  \exp(x_2 - x_3) -\exp x_2 - 0.05 \exp(x_1 + x_3) \geq 0\\
                                        &~ g_{2:4}(x) \doteq \exp(x) - (70,\,1,\, 0.5) \geq (0, 0, 0)  \\
                                        &~ g_{5:7}(x) \doteq (150,\,30,\,21) - \exp(x) \geq (0, 0, 0)
    \end{align*}

It's often easier to specify a signomial program by defining symbols ``y``,
which are related to variables ``x`` by ``y = exp(x)``. You can get a hold of these symbols ``y`` by using the
function ``standard_sig_monomials``, and providing a dimension of your desired variable. ::

    import sageopt as so
    y = so.standard_sig_monomials(3)
    f = 0.5 * y[0] * y[1] ** -1 - y[0] - 5 * y[1] ** -1
    gts = [100 -  y[1] * y[2] ** -1 - y[1] - 0.05 * y[0] * y[2],
           y[0] - 70,
           y[1] - 1,
           y[2] - 0.5,
           150 - y[0],
           30 - y[1],
           21 - y[2]]
    eqs = []

Next we will pass our problem through a function called ``conditional_sage_data``. This function parses the given
constraint signomials, and infers any which can be written in a tractable convex form with respect to the
optimization variable ``x``
. ::

  X = so.conditional_sage_data(f, gts, eqs)

For this problem, it just so happens that all constraints can be written in a convex form. Taking this as given, we use
the function ``sig_relaxation``
for the resulting problem. The dual formulation is used because we want to recover a solution. ::

    dual = so.sig_relaxation(f, X=X, form='dual')
    dual.solve(verbose=False)
    solutions = so.sig_solrec(dual)
    best_soln = solutions[0]
    print(best_soln)

Now let's see if this solution is any good! ::

    print("The recovered solution has objective value ...")
    print('\t' + str(f(best_soln)))  # about -147.66666
    print("The recovered solution has constraint violation ...")
    constraint_levels = min([g(best_soln) for g in gts])  # zero!
    violation = 0 if constraint_levels >= 0 else -constraint_levels
    print('\t' + str(violation))
    print('The level 0 SAGE bound is ... ')
    print('\t' + str(dual.value))  # about -147.857

We can certify that the solution is actually much closer to optimality than the SAGE bound would suggest. We can
easily construct and solve a level ``ell=3`` SAGE relaxation to produce a stronger lower bound on this minimization
problem. ::

    dual = so.sig_relaxation(f, X=X, form='dual', ell=3)
    dual.solve(verbose=False)
    print('The level 3 SAGE bound is ... ')
    print('\t' + str(dual.value))  # about  -147.6666

Example 2 (signomial)
~~~~~~~~~~~~~~~~~~~~~

We want to solve the following equality-constrained signomial program.

.. math::

   \begin{align*}
    \min_{\substack{A \in \mathbb{R}^3_{++} \\ P \in \mathbb{R}_{++} }} 
               &~ 10^4 (A_1 + A_2 + A_3)  \\
   \text{s.t.} &~ 10^4 + 0.01 A_1^{-1}A_3^{} - 7.0711 A_1^{-1} \geq 0 \\
               &~ 10^4 + 0.00854 A_1^{-1}P - 0.60385(A_1^{-1} + A_2^{-1}) \geq 0  \\
               &~ 70.7107 A_1^{-1} - A_1^{-1}P - A_{3}^{-1}P = 0 \\
               &~ 10^4 \geq 10^4 A_1 \geq 10^{-4} \qquad  10^4 \geq 10^4 A_2 \geq 7.0711 \\
               &~ 10^4 \geq 10^4 A_3 \geq 10^{-4} \qquad 10^4 \geq 10^4 P_{~} \geq 10^{-4} 
   \end{align*}


It is straightforward to compute a tight bound on the problem's optimal objective, however solution recovery is
difficult. Thus we show this problem in two forms: once with the equality constraint, and once where the inequality
constraint is used to *define* a value of :math:`P` (which we can then substitute into the rest of the formulation).
First we show the case with the equality constraint. ::

   import sageopt as so
   x = so.standard_sig_monomials(4)
   A = x[:3]
   P = x[3]
   f = 1e4 * sum(A)
   main_gts = [
       1e4 + 1e-2 * A[2] / A[0] - 7.0711 / A[0],
       1e4  + 8.54e-3 * P/ A[0] - 6.0385e-1 * (1.0 / A[0] + 1.0 / A[1])
   ]
   bounds = [
       1e4 - 1e4 * A[0], 1e4 * A[0] - 1e-4,
       1e4 - 1e4 * A[1], 1e4 * A[1] - 7.0711,
       1e4 - 1e4 * A[2], 1e4 * A[2] - 1e-4,
       1e4 - 1e4 * P, 1e4 * P - 1e-4
   ]
   gts = main_gts + bounds
   eqs = [70.7107 / A[0] + P / A[0] - P / A[2]]
   X = so.conditional_sage_data(f, bounds, [])
   prim = so.sig_constrained_relaxation(f, main_gts, eqs, form='primal', X=X)
   dual = so.sig_constrained_relaxation(f, main_gts, eqs, form='dual', X=X)
   prim.solve(verbose=False)
   dual.solve(verbose=False)
   print('\n')
   print(prim.value)
   print(dual.value)

The equality constraint in this problem creates an unnecessary challenge in solution recovery. Since we usually want
to recover optimal solutions, we reformulate the problem by substituting :math:`P \leftarrow 70.7107 A_3 / (A_1 + A_3)
`, and clearing the denominator :math:`(A_1 + A_3)` from constraints which involved :math:`P`. ::

   A = so.standard_sig_monomials(3)
   f = 1e4 * sum(A)
   main_gts = [
       1e4 + 1e-2 * A[2] / A[0] - 7.0711 / A[0],
       1e4 * (A[2] + A[0]) + 8.54e-3 * (70.7012 * A[2] * (A[0] + A[2])) / A[0]
           - 6.0385e-1 * (A[0] + A[2]) * (1.0 / A[0] + 1.0 / A[1])
   ]
   bounds = [
       1e4 - 1e4 * A[0], 1e4 * A[0] - 1e-4,
       1e4 - 1e4 * A[1], 1e4 * A[1] - 7.0711,
       1e4 - 1e4 * A[2], 1e4 * A[2] - 1e-4,
       A[0] - 69.7107 * A[2], (1e8 * 70.7107 - 1) - A[0] / A[2]
   ]
   gts = main_gts + bounds
   X = so.conditional_sage_data(f, gts, [])
   dual = so.sig_constrained_relaxation(f, main_gts, [], form='dual', X=X)
   dual.solve()
   print('\n')
   print(dual.value)
   solns = so.sig_solrec(dual)
   print(f(solns[0]))


Example 3 (polynomial)
~~~~~~~~~~~~~~~~~~~~~~

In this example, we minimize

.. math::

   f(x) = -64 \sum_{i=1}^7 \prod_{j \neq i} x_j

over :math:`x \in [-1/2, 1/2]^7`. We also want to recover optimal solutions. ::

   import numpy as np
   import sageopt as so
   x = so.standard_poly_monomials(7)
   f = 0
   for i in range(7):
       sel = np.ones(7, dtype=bool)
       sel[i] = False
       f -= 64 * np.prod(x[sel])
       # ^ use simple NumPy functions to construct Polynomials!
   gts = [0.25 - x[i]**2 for i in range(7)]  # -.5 <= x[i] <= .5
   X = so.conditional_sage_data(f, gts, [])
   dual = so.poly_constrained_relaxation(f, [], [], form='dual', X=X)
   dual.solve(verbose=False, solver='MOSEK')
   print()
   solns = so.poly_solrec(dual)
   for sol in solns:
       print(sol)

You can also try this example with ECOS. When using ECOS, you might want to use local solver refinement, as accessed
in ``sageopt.local_refinement``.

Example 4 (polynomial)
~~~~~~~~~~~~~~~~~~~~~~

We want to solve a degree six polynomial optimization problem in six variables.

.. math::
   \begin{align*}
    \min_{x \in \mathbb{R}^6} &~ f(x) \doteq x_1^6 - x_2^6 + x_3^6 - x_4^6 + x_5^6 - x_6^6 + x_1 - x_2 \\
                           \text{s.t.} &~ g_1(x) \doteq 2 x_{1}^{6}+3 x_{2}^{2}+2 x_{1} x_{2}+2 x_{3}^{6}+3 x_{4}^{2}+2 x_{3} x_{4}+2 x_{5}^{6}+3 x_{6}^{2}+2 x_{5} x_{6}  \geq 0  \\
                                       &~ g_2(x) \doteq 2 x_{1}^{2}+5 x_{2}^{2}+3 x_{1} x_{2}+2 x_{3}^{2}+5 x_{4}^{2}+3 x_{3} x_{4}+2 x_{5}^{2}+5 x_{6}^{2}+3 x_{5} x_{6}  \geq 0 \\
                                       &~ g_3(x) \doteq 3 x_{1}^{2}+2 x_{2}^{2}-4 x_{1} x_{2}+3 x_{3}^{2}+2 x_{4}^{2}-4 x_{3} x_{4}+3 x_{5}^{2}+2 x_{6}^{2}-4 x_{5} x_{6} \geq 0  \\
                                       &~ g_4(x) \doteq x_{1}^{2}+6 x_{2}^{2}-4 x_{1} x_{2}+x_{3}^{2}+6 x_{4}^{2}-4 x_{3} x_{4}+x_{5}^{2}+6 x_{6}^{2}-4 x_{5} x_{6} \geq 0 \\
                                       &~ g_5(x) \doteq x_{1}^{2}+4 x_{2}^{6}-3 x_{1} x_{2}+x_{3}^{2}+4 x_{4}^{6}-3 x_{3} x_{4}+x_{5}^{2}+4 x_{6}^{6}-3 x_{5} x_{6} \geq 0 \\
                                       &~ g_{6:10}(x) \doteq 1 - g_{1:5}(x) \geq  (0, 0, 0, 0, 0) \\
                                       &~ g_{11:16}(x) = x \geq (0, 0, 0, 0, 0, 0)
   \end{align*}


The ``sageopt`` approach to this problem is to write it first as a signomial program, and then perform solution recovery with consideration to the underlying polynomial structure. The solution recovery starts with ``sig_solrec`` as normal, but then we refine the solution with a special function ``local_refine_polys_from_sigs``. ::

   import sageopt as so

   x = so.standard_sig_monomials(6)
   f = x[0]**6 - x[1]**6 + x[2]**6 - x[3]**6 + x[4]**6 - x[5]**6 + x[0] - x[1]

   expr1 = 2*x[0]**6 + 3*x[1]**2 + 2*x[0]*x[1] + 2*x[2]**6 + 3*x[3]**2 + 2*x[2]*x[3] + 2*x[4]**6 + 3*x[5]**2 + 2*x[4]*x[5]
   expr2 = 2*x[0]**2 + 5*x[1]**2 + 3*x[0]*x[1] + 2*x[2]**2 + 5*x[3]**2 + 3*x[2]*x[3] + 2*x[4]**2 + 5*x[5]**2 + 3*x[4]*x[5]
   expr3 = 3*x[0]**2 + 2*x[1]**2 - 4*x[0]*x[1] + 3*x[2]**2 + 2*x[3]**2 - 4*x[2]*x[3] + 3*x[4]**2 + 2*x[5]**2 - 4*x[4]*x[5]

   expr4 = x[0]**2 + 6*x[1]**2 - 4*x[0]*x[1] + x[2]**2 + 6*x[3]**2 - 4*x[2]*x[3] + x[4]**2 + 6*x[5]**2 - 4*x[4]*x[5]
   expr5 = x[0]**2 + 6*x[1]**2 - 4*x[0]*x[1] + x[2]**2 + 6*x[3]**2 - 4*x[2]*x[3] + x[4]**2 + 6*x[5]**2 - 4*x[4]*x[5]

   gts = [expr3, expr4, expr5, 1 - expr1, 1 - expr2, 1 - expr3, 1 - expr4, 1 - expr5]
   eqs = []

   dual = so.sig_constrained_relaxation(f, gts, eqs, form='dual', p=1, q=1, ell=0)
   dual.solve(verbose=False, solver='MOSEK')  # ECOS fails
   y0 = so.sig_solrec(dual)[0]
   x_star = so.local_refine_polys_from_sigs(f, gts, eqs, y0)

   print()
   print(dual.value)
   f_poly = f.as_polynomial()
   print(f_poly(x_star))
   print(x_star)


Nonnegativity Examples
----------------------

Although sageopt is designed around optimization, the mechanism by which sageopt operates is to certify nonnegativity
by decomposing a given function into a "Sum of AGE-functions". These AGE functions are nonnegative, and can be proven
nonnegative in a relatively simple way. If you want to check nonnegativity of the AGE functions yourself (you might
find yourself in this situation if a numerical solver seemed to struggle with a SAGE relaxation), then you can do
that. Here we show how to get a hold on these AGE functions, from a given SAGE relaxation.

Example 1 (signomial)
~~~~~~~~~~~~~~~~~~~~~

Consider the following optimization problem:

.. math::

    \begin{align*}
     \min_{x \in \mathbb{R}^3} &~ f(x) \doteq 0.5 \exp(x_1 - x_2) -\exp x_1  - 5 \exp(-x_2)  \\
                            \text{s.t.} &~ g_1(x) \doteq 100 -  \exp(x_2 - x_3) -\exp x_2 - 0.05 \exp(x_1 + x_3) \geq 0\\
                                        &~ g_{2:4}(x) \doteq \exp(x) - (70,\,1,\, 0.5) \geq (0, 0, 0)  \\
                                        &~ g_{5:7}(x) \doteq (150,\,30,\,21) - \exp(x) \geq (0, 0, 0)
    \end{align*}

We can produce a bound on this minimum with a primal SAGE relaxation. ::

   import sageopt as so
   y = so.standard_sig_monomials(3)
   f = 0.5 * y[0] * y[1] ** -1 - y[0] - 5 * y[1] ** -1
   gts = [100 -  y[1] * y[2] ** -1 - y[1] - 0.05 * y[0] * y[2],
          y[0] - 70, y[1] - 1, y[2] - 0.5,
          150 - y[0], 30 - y[1], 21 - y[2]]
   X = so.conditional_sage_data(f, gts, [])
   prim = so.sig_relaxation(f, form='primal', ell=0, X=X)
   prim.solve(solver='ECOS')
   print(prim.value)  # about -147.857

As long as the solver (here, ECOS) succeeds in solving the problem, the function ``f - prim.value`` should be
nonnegative over the set represented by ``X``. The intended proof that ``f - prim.value`` is nonnegative comes from
the AGE functions participating in its decomposition. We can recover those functions as follows ::

   sage_constraint = prim.user_cons[0]  # a PrimalSageCone object
   alpha = sagecon.alpha
   agefunctions = []
   for ci in sagecon.age_vectors.values():
       s = so.Signomial(alpha, ci.value)
       agefunctions.append(s)

You should find that one of these AGE functions has very small positive coefficients, and a large negative term. We can
investigate this suspicious AGE function further. Specifically, we can transform the suspicious AGE function into a
convex function, and then solve a constrained convex optimization problem using a function from ``scipy``. ::

   suspicious_age = agefunctions[1]
   convexified_suspicious_age = y[1] * suspicious_age
   import numpy as np
   from scipy.optimize import fmin_cobyla
   def sample_initial_point():
       y1 = 70 + 80 * np.random.rand()
       y2 = 1 + 29 * np.random.rand()
       y3 = 0.5 + 20.5 * np.random.rand()
       x0 = np.log([y1, y2, y3])
       return x0
   fmin_cobyla(convexified_suspicious_age,
               sample_initial_point(), gts,
               disp=1, maxfun=1e5, rhoend=1e-7)

You should find that no matter how many initial conditions you provide to ``scipy``'s solver, the reported optimal
objective is nonnegative.
