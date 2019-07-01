.. toctree::
   :maxdepth: 2

Examples from MCW19
===================

The examples shown here appear in the 2019 article by Murray, Chandrasekaran, and Wierman, titled *Signomial and
Polynomial Optimization via Relative Entropy and Partial Dualization*. That paper introduced conditional SAGE
certificates.

This page is under construction!

Example 1 (signomial)
---------------------

We want to solve a constraint signomial program in three variables.

.. math::

    \begin{align*}
     \min_{x \in \mathbb{R}^3} &~ f(x) \doteq 0.5 \exp(x_1 - x_2) -\exp x_1  - 5 \exp(-x_2)  \\
                            \text{s.t.} &~ g_1(x) \doteq 100 -  \exp(x_2 - x_3) -\exp x_2 - 0.05 \exp(x_1 + x_3) \geq 0\\
                                        &~ g_{2:4}(x) \doteq \exp(x) - (70,\,1,\, 0.5) \geq (0, 0, 0)  \\
                                        &~ g_{5:7}(x) \doteq (150,\,30,\,21) - \exp(x) \geq (0, 0, 0)
    \end{align*}

It's often easier to specify a signomial program by defining symbols ``y``,
which are related to variables ``x`` by ``y = exp(x)``. ::

    from sageopt import conditional_sage_data, sig_dual
    from sageopt import standard_sig_monomials, sig_solrec
    n = 3
    y = standard_sig_monomials(n)
    f = 0.5 * y[0] * y[1] ** -1 - y[0] - 5 * y[1] ** -1
    gts = [100 -  y[1] * y[2] ** -1 - y[1] - 0.05 * y[0] * y[2],
           y[0] - 70,
           y[1] - 1,
           y[2] - 0.5,
           150 - y[0],
           30 - y[1],
           21 - y[2]]
    eqs = []
    X = conditional_sage_data(f, gts, eqs)

We will use ``sig_dual`` for this problem. The dual formulation is used because we want to recover a solution,
rather than just produce a bound on the optimization problem. We begin by solving a level ``ell=0`` relaxation. ::

    dual = sig_dual(f, ell=0, X=X)
    dual.solve(verbose=False)
    solutions = sig_solrec(dual)
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

    dual = sig_dual(f, ell=3, X=X)
    dual.solve(verbose=False)
    print('The level 3 SAGE bound is ... ')
    print('\t' + str(dual.value))  # about  -147.6666

Example 2 (signomial)
---------------------

under construction!

Example 3 (polynomial)
----------------------

under construction!

Example 4 (polynomial)
----------------------

under construction!

Example 5 (signomial)
---------------------

under construction!

Example 6 (signomial)
---------------------

under construction!

Example 7 (signomial)
---------------------

under construction!

Example 8 (signomial)
---------------------

under construction!

Example 9 (signomial)
---------------------

under construction!

Example 10 (polynomial)
-----------------------

Under construction!

Example 11 (polynomial)
-----------------------

Under construction!

Example 12 (polynomial)
-----------------------

Under construction!



