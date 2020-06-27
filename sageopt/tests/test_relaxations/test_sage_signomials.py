"""
   Copyright 2018 Riley John Murray

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import unittest
import numpy as np
from sageopt import coniclifts as cl
from sageopt.coniclifts.constraints.set_membership import sage_cones
from sageopt.relaxations import sig_relaxation, sig_constrained_relaxation, sage_multiplier_search
from sageopt.relaxations import sig_solrec, infer_domain
from sageopt.symbolic.signomials import Signomial, standard_sig_monomials


def primal_dual_vals(f, ell, X=None, solver='ECOS'):
    # primal
    prob = sig_relaxation(f, X, form='primal', ell=ell)
    status, value = prob.solve(solver=solver, verbose=False)
    prim = value
    # dual
    prob = sig_relaxation(f, X, form='dual', ell=ell)
    status, value = prob.solve(solver=solver, verbose=False)
    dual = value
    return [prim, dual], prob


def constrained_primal_dual_vals(f, gts, eqs, p, q, ell, X, solver='ECOS'):
    # primal
    prob = sig_constrained_relaxation(f, gts, eqs,
                                      form='primal', p=p, q=q, ell=ell, X=X)
    status, value = prob.solve(solver=solver, verbose=False)
    prim = value
    # dual
    prob = sig_constrained_relaxation(f, gts, eqs,
                                      form='dual', p=p, q=q, ell=ell, X=X)
    status, value = prob.solve(solver=solver, verbose=False)
    dual = value
    return [prim, dual], prob


# noinspection SpellCheckingInspection
class TestSAGERelaxations(unittest.TestCase):

    def test_unconstrained_sage_1(self, presolve=False, compactdual=True, kernel_basis=False):
        # Background
        #
        #       This is Example 1 from a 2018 paper by Murray, Chandrasekaran, and Wierman
        #       (https://arxiv.org/pdf/1810.01614.pdf).
        #
        # Tests
        #
        #       (1) Check that primal / dual objectives are close to a reference values, for ell \in {0, 1}.
        #
        #       (2) Recover a globally optimal solution at ell == 1.
        #
        initial_presolve = sage_cones.SETTINGS['presolve_trivial_age_cones']
        initial_compactdual = sage_cones.SETTINGS['compact_dual']
        initial_kb = sage_cones.SETTINGS['kernel_basis']
        cl.presolve_trivial_age_cones(presolve)
        cl.compact_sage_duals(compactdual)
        cl.kernel_basis_age_witnesses(kernel_basis)
        alpha = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 0],
                          [0, 0.5]])
        c = np.array([0, 3, 2, 1, -4, -2])
        s = Signomial(alpha, c)
        expected = [-1.83333, -1.746505595]
        pd0, _ = primal_dual_vals(s, 0)
        self.assertAlmostEqual(pd0[0], expected[0], 4)
        self.assertAlmostEqual(pd0[1], expected[0], 4)
        pd1, dual = primal_dual_vals(s, 1)
        self.assertAlmostEqual(pd1[0], expected[1], 4)
        self.assertAlmostEqual(pd1[1], expected[1], 4)
        optsols = sig_solrec(dual)
        assert (s(optsols[0]) - dual.value) < 1e-6
        cl.presolve_trivial_age_cones(initial_presolve)
        cl.compact_sage_duals(initial_compactdual)
        cl.kernel_basis_age_witnesses(initial_kb)

    def test_unconstrained_sage_1a(self):
        self.test_unconstrained_sage_1(True, True, False)

    def test_unconstrained_sage_1b(self):
        self.test_unconstrained_sage_1(False, False, True)

    def test_unconstrained_sage_2(self):
        # Background
        #
        #       This is Example 2 from a 2018 paper by Murray, Chandrasekaran, and Wierman
        #       (https://arxiv.org/pdf/1810.01614.pdf).
        #
        # Tests
        #
        #       (1) Check that primal / dual objective are -\infty for ell == 0.
        #
        #       (2) Check that primal / dual objectives are close to a reference value, for ell == 1.
        #
        #       (3) Recover a globally optimal solution at ell == 1.
        #
        alpha = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 1],
                          [1, 0.5]])
        c = np.array([0, 1, 1, 1.9, -2, -2])
        s = Signomial(alpha, c)
        expected = [-np.inf, -0.122211863]
        pd0, _ = primal_dual_vals(s, 0)
        assert pd0[0] == expected[0] and pd0[1] == expected[0]
        pd1, dual = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-5 and abs(pd1[1] - expected[1]) < 1e-5
        solns = sig_solrec(dual)
        assert s(solns[0]) < 1e-6 + dual.value

    def test_unconstrained_sage_3(self):
        # Background
        #
        #       This is Example 2.5 from the original SAGE paper by Chandrasekaran and Shah.
        #       The signomial s(x1,x2,x3) = (exp(x1) - exp(x2) - exp(x3))**2 is nonnegative
        #       over R^3, but it is not SAGE.
        #
        # Tests
        #
        #       (1) Show that the standard SAGE hierarchy produces no finite bound on "s",
        #           for ell \in {0, 1}.
        #
        # Notes
        #
        #       It is suspected that the standard SAGE hierarchy never produces a finite bound
        #       for this signomial.
        #
        s = Signomial.from_dict({(1, 0, 0): 1,
                                 (0, 1, 0): -1,
                                 (0, 0, 1): -1})
        s = s ** 2
        expected = -np.inf
        pd0, _ = primal_dual_vals(s, 0)
        assert pd0[0] == expected and pd0[1] == expected
        pd1, _ = primal_dual_vals(s, 1)
        assert pd1[0] == expected and pd1[1] == expected

    def test_unconstrained_sage_4(self, presolve=False, compactdual=False, kernel_basis=False):
        # Background
        #
        #       This example was constructed soley as a test case for sageopt.
        #
        #       Minimize s(x) = exp(3*x) - 4*exp(2*x) + 7*exp(x) + exp(-x), over x \in R.
        #
        # Tests
        #
        #       (1) Check that primal / dual objectives are close to reference values, for ell \in {0, 1, 2}.
        #
        #       (2) Recover a globally optimal solution from the dual relaxation, when ell == 3.
        #
        # Notes
        #
        #       It may not be obvious, but the signomial "s" is actually convex!
        #
        initial_presolve = sage_cones.SETTINGS['presolve_trivial_age_cones']
        initial_compactdual = sage_cones.SETTINGS['compact_dual']
        initial_kb = sage_cones.SETTINGS['kernel_basis']
        cl.presolve_trivial_age_cones(presolve)
        cl.compact_sage_duals(compactdual)
        cl.kernel_basis_age_witnesses(kernel_basis)
        s = Signomial.from_dict({(3,): 1, (2,): -4, (1,): 7, (-1,): 1})
        expected = [3.464102, 4.60250026, 4.6217973]
        pds = [primal_dual_vals(s, ell) for ell in range(3)]
        for ell in range(3):
            assert abs(pds[ell][0][0] - expected[ell]) < 1e-5
            assert abs(pds[ell][0][1] - expected[ell]) < 1e-5
        dual = sig_relaxation(s, form='dual', ell=3)
        dual.solve(solver='ECOS', verbose=False)
        optsols = sig_solrec(dual)
        assert s(optsols[0]) < 1e-6 + dual.value
        cl.presolve_trivial_age_cones(initial_presolve)
        cl.compact_sage_duals(initial_compactdual)
        cl.kernel_basis_age_witnesses(initial_kb)

    def test_unconstrained_sage_4a(self):
        self.test_unconstrained_sage_4(True, False, False)

    def test_unconstrained_sage_4b(self):
        self.test_unconstrained_sage_4(False, True, True)

    def test_unconstrained_sage_5(self):
        # Background
        #
        #       This is Example 4 from a 2018 paper by Murray, Chandrasekaran, and Wierman
        #       (https://arxiv.org/pdf/1810.01614.pdf).
        #
        # Tests
        #
        #       (1) check that primal / dual objectives are close to reference values, for ell \in {0, 1}.
        #
        alpha = np.array([[0., 1.],
                          [0.21, 0.08],
                          [0.16, 0.54],
                          [0., 0.],
                          [1., 0.],
                          [0.3, 0.58]])
        c = np.array([1., -57.75, -40.37, 33.94, 67.29, 38.28])
        s = Signomial(alpha, c)
        expected = [-24.054866, -21.31651]
        pd0, _ = primal_dual_vals(s, 0)
        assert abs(pd0[0] - expected[0]) < 1e-4 and abs(pd0[1] - expected[0]) < 1e-4
        pd1, _ = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-4 and abs(pd1[1] - expected[1]) < 1e-4

    def test_unconstrained_sage_6(self):
        # Background
        #
        #       This is Example 5 from a 2018 paper by Murray, Chandrasekaran, and Wierman
        #       (https://arxiv.org/pdf/1810.01614.pdf).
        #
        # Tests
        #
        #       (1) check that primal / dual objectives are close to reference values, for ell \in {0, 1}.
        #
        alpha = np.array([[0., 1.],
                         [0., 0.],
                         [0.52, 0.15],
                         [1., 0.],
                         [2., 2.],
                         [1.3, 1.38]])
        c = np.array([2.55, 0.31, -1.48, 0.85, 0.65, -1.73])
        s = Signomial(alpha, c)
        expected = [0.00354263, 0.13793126]
        pd0, _ = primal_dual_vals(s, 0)
        assert abs(pd0[0] - expected[0]) < 1e-6 and abs(pd0[1] - expected[0]) < 1e-6
        pd1, _ = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-6 and abs(pd1[1] - expected[1]) < 1e-6

    def test_sage_multiplier_search(self):
        # Background
        #
        #       This example was constructed solely as a test case for sageopt.
        #
        #       The problem is to find a bound on the nonnegative signomial
        #       s(x) = (exp(x)  - exp(-x))**4, using the machinery of SAGE certificates.
        #
        # Tests
        #
        #       (1) Show that there is no SAGE signomial "f" (over the same exponents as "s")
        #           such that f * s is SAGE.
        #
        #       (2) Obtain a loose (but finite) bound on "s", via a SAGE relaxation with ell == 1.
        #
        #       (3) Improve the finite bound from Test 2 by verifying nonnegativity of an
        #           appropriate translate of "s".
        #
        s = Signomial.from_dict({(1,): 1, (-1,): -1}) ** 4
        prob0 = sage_multiplier_search(s, level=1)
        res0 = prob0.solve(solver='ECOS', verbose=False)
        val0 = res0[1]
        assert val0 == -np.inf
        prob1 = sig_relaxation(s, form='primal', ell=1)
        res1 = prob1.solve(solver='ECOS', verbose=False)
        s_bound = res1[1]
        assert -np.inf < s_bound < 0
        s_shifted = s - 0.5 * s_bound  # shifted_s is nonnegative, and not-SAGE by construction.
        prob2 = sage_multiplier_search(s_shifted, level=1)
        res2 = prob2.solve(solver='ECOS', verbose=False)
        val2 = res2[1]
        assert val2 == 0.

    @staticmethod
    def _constrained_sage_1():
        # Background
        #
        #       This is Example 3.3 from Chandraskearan and Shah's original paper on SAGE relaxations.
        #       The problem is to minimize a nonconvex signomial, over a convex set defined by a single
        #       posynomial inequality.
        #
        s0 = Signomial.from_dict({(10.2, 0, 0): 10, (0, 9.8, 0): 10, (0, 0, 8.2): 10})
        s1 = Signomial.from_dict({(1.5089, 1.0981, 1.3419): -14.6794})
        s2 = Signomial.from_dict({(1.0857, 1.9069, 1.6192): -7.8601})
        s3 = Signomial.from_dict({(1.0459, 0.0492, 1.6245): 8.7838})
        f = s0 + s1 + s2 + s3
        g = Signomial.from_dict({(10.2, 0, 0): -8,
                                 (0, 9.8, 0): -8,
                                 (0, 0, 8.2): -8,
                                 (1.0857, 1.9069, 1.6192): -6.4,
                                 (0, 0, 0): 1})
        gs = [g]
        return f, gs

    def test_constrained_sage_1a(self):
        # Tests - (p, q, ell) = (0, 1, 0)
        #
        #       (1) Verify that primal and dual objectives are close to a reference value.
        #
        #       (2) Recover a solution (feasible up to tol 1e-7) with at most 0.01 percent optimality gap
        #
        f, gs = TestSAGERelaxations._constrained_sage_1()
        expected = -0.6147
        actual, dual = constrained_primal_dual_vals(f, gs, [], p=0, q=1, ell=0, X=None)
        assert abs(actual[0] - expected) < 1e-4 and abs(actual[1] - expected) < 1e-4
        solns = sig_solrec(dual, ineq_tol=1e-7)
        assert (f(solns[0]) - dual.value) / abs(dual.value) < 1e-4

    def test_constrained_sage_1b(self):
        # Tests - (p, q, ell) = (0, 1, 0)
        #
        #       (1) Solve the dual with and without introduction of additional slack variables,
        #           verify that objectives are similar.
        #
        f, gs = TestSAGERelaxations._constrained_sage_1()
        dual1 = sig_constrained_relaxation(f, gs, [], slacks=False)
        dual2 = sig_constrained_relaxation(f, gs, [], slacks=True)
        self.assertLessEqual(dual1.A.shape[0], dual2.A.shape[0])
        dual1.solve(verbose=False)
        dual2.solve(verbose=False)
        self.assertAlmostEqual(dual1.value, dual2.value, places=3)

    def test_constrained_sage_2(self):
        # Background
        #
        #       This is a signomial formulation of a nonnegative polynomial optimization problem.
        #
        #       The problem can be found on page 16 of the gloptipoly3 manual
        #                   http://homepages.laas.fr/henrion/papers/gloptipoly3.pdf
        #       among other places. The optimal objective is -4.
        #
        # Tests - (p, q, ell) = (0, 1, 0)
        #
        #       (1) Check for similar primal / dual objectives.
        #
        x = standard_sig_monomials(3)
        f = -2 * x[0] + x[1] - x[2]
        g1 = Signomial.from_dict({(0, 0, 0): 24,
                        (1, 0, 0): -20,
                        (0, 1, 0): 9,
                        (0, 0, 1): -13,
                        (2, 0, 0): 4,
                        (1, 1, 0): -4,
                        (1, 0, 1): 4,
                        (0, 2, 0): 2,
                        (0, 1, 1): -2,
                        (0, 0, 2): 2})
        g2 = 4 - x[0] - x[1] - x[2]
        g3 = 6 - 3*x[1] - x[2]
        g4 = 2 - x[0]
        g5 = 3 - x[2]
        gts = [g1, g2, g3, g4, g5]
        res01, _ = constrained_primal_dual_vals(f, gts, [], p=0, q=1, ell=0, X=None)
        expect = -6
        assert abs(res01[0] - expect) < 1e-4
        assert abs(res01[1] - expect) < 1e-4
        assert abs(res01[0] - res01[1]) < 1e-5

    def test_conditional_sage_1(self):
        # Background
        #
        #       This is Problem 1 from MCW2019. It appears in many papers on algorithms
        #       for signomial progamming. All constraints are convex.
        #
        # Tests
        #
        #       ell=0 and ell=3. Test for recovery of a feasible solution for
        #       ell=0, and test for recovery of an optimal solution for ell=3.
        #
        # Notes
        #
        #       ECOS can't solve the primal when ell > 0. Only try ell=3 if MOSEK is installed.
        #
        y = standard_sig_monomials(3)
        f = 0.5 * y[0] * y[1] ** -1 - y[0] - 5 * y[1] ** -1
        gts = [100 - y[1] * y[2] ** -1 - y[1] - 0.05 * y[0] * y[2],
               y[0] - 70, y[1] - 1, y[2] - 0.5,
               150 - y[0], 30 - y[1], 21 - y[2]]
        eqs = []
        X = infer_domain(f, gts, eqs)
        vals, prob = primal_dual_vals(f, 0, X)
        expect = -147.85712
        assert abs(vals[0] - expect) <= 1e-3
        assert abs(vals[1] - expect) <= 1e-3
        solutions = sig_solrec(prob)
        assert len(solutions) > 0
        if cl.Mosek.is_installed():
            vals, prob = primal_dual_vals(f, 3, X, solver='MOSEK')
            expect = -147.6666
            assert abs(vals[0] - expect) <= 1e-3
            assert abs(vals[1] - expect) <= 1e-3
            solutions = sig_solrec(prob)
            assert len(solutions) > 0
            x_star = solutions[0]
            gap = f(x_star) - prob.value
            assert abs(gap) <= 1e-3
        pass

    def test_conditional_constrained_sage_1(self):
        # Background
        #
        #       This is Problem 1 from a 2014 paper by Xue-Ping Hou, Pei-Ping Shen, and Yong-Qiang Chen.
        #       It can also be found in * many * other papers.
        #       The optimal objective is 0.76508
        #
        # Tests - (p, q, ell) = (0, 1, 0).
        #
        #       (1) Check that primal and dual objectives are close.
        #
        #       (2) Check that primal objective is close to reference value.
        #
        #       (3) Verify that we can recover an optimal solution.
        #
        n = 4
        y = standard_sig_monomials(n)
        f = y[2] ** 0.8 * y[3] ** 1.2
        gts = [y[0] * y[3] ** -1 + y[1] ** -1 * y[3] ** -1 - 1,
               - y[0] ** -2 * y[2] ** -1 - y[1] * y[2] ** -1 + 1]
        gts = [-g for g in gts]
        gts += [1 - y[0], y[0] - 0.1,
                10 - y[1], y[1] - 5,
                15 - y[2], y[2] - 8,
                1 - y[3], y[3] - 0.01]
        eqs = []
        X = infer_domain(f, gts, eqs)
        p, q, ell = 0, 1, 0
        vals, dual = constrained_primal_dual_vals(f, gts, eqs, p, q, ell, X)
        assert abs(vals[0] - vals[1]) < 1e-5
        assert abs(vals[0] - 0.765082) < 1e-4
        solns = sig_solrec(dual, ineq_tol=0)
        assert f(solns[0]) < 1e-8 + dual.value

    @unittest.skipUnless(cl.Mosek.is_installed(), 'ECOS takes too long for this problem.')
    def test_conditional_constrained_sage_2(self):
        # Background
        #
        #       This is Problem 4 from a 2005 paper by Yanjun Wang and Zhian Liang.
        #       The optimal objective is between 11.95 and 11.96.
        #
        # Tests - (p, q, ell) = (0, 2, 0)
        #
        #       (1) Verify similar primal / dual objectives.
        #
        #       (2) Verify that primal objective is within 1 percent of a reference value.
        #
        #       (3) Recover a solution feasible within 1e-8, within 1 percent of optimality.
        #
        n = 2
        y = standard_sig_monomials(n)
        f = 3.7 * y[0] ** 0.85 + 1.985 * y[0] + 700.3 * y[1] ** -0.75
        gts = [1 - 0.7673 * y[1] ** 0.05 + 0.05 * y[0],
               5 - y[0],   y[0] - 0.1,
               450 - y[1], y[1] - 380]
        eqs = []
        X = infer_domain(f, gts, eqs)
        p, q, ell = 0, 2, 0
        vals, dual = constrained_primal_dual_vals(f, gts, eqs, p, q, ell, X, solver='MOSEK')
        assert abs(vals[0] - vals[1]) < 1e-1
        assert abs(vals[0] - 11.95) / vals[0] < 1e-2
        solns = sig_solrec(dual, ineq_tol=1e-8)
        assert (f(solns[0]) - dual.value) / dual.value < 1e-2

    @unittest.skipUnless(cl.Mosek.is_installed(), 'ECOS takes too long for this problem.')
    def test_conditional_constrained_sage_3(self):
        # Background
        #
        #       This is a modification of Problem 10 from the 1978 paper by M. Rijckaert and X. Martens.
        #       The bound constraints come from a different paper (but I dont recall where...).
        #       We also add an additional trivially-valid constraint (simply to strengthen the Lagrange dual).
        #       The optimal objective is reported in the literature as approximately -83.21.
        #
        # Tests - (p, q, ell) = (0, 1, 1)
        #
        #       (1) Verify similar primal / dual objectives.
        #
        #       (2) Recover a strictly feasible solution, within 0.7 % of optimality.
        #
        # Notes
        #
        #       With (p, q, ell) = (0, 0, 2) (not shown here), we get a SAGE bound of -83.253.
        #       If the (0,0,2) SAGE bound of -83.253 is to be believed, then the recovered solution
        #       actually has a relative optimality gap of only 0.003 percent.
        #
        n = 3
        x = standard_sig_monomials(n)
        f = 0.5 * x[0] * (x[1] ** -1) - x[0] - 5.0 * (x[1] ** -1)
        g = 1 - 0.01 * x[1] * (x[2] ** -1) - 0.01 * x[0] - 0.0005 * x[0] * x[2]
        g = 100.0 * g
        gts = [g, g * (x[1] ** -2),
               1e2 - x[0], 1e2 - x[1], 1e2 - x[2],
               x[0] - 1, x[1] - 1, x[2] - 1]
        eqs = []
        X = infer_domain(f, gts, eqs)
        p, q, ell = 0, 1, 1
        vals, dual = constrained_primal_dual_vals(f, gts, eqs, p, q, ell, X, solver='MOSEK')
        assert abs(vals[0] - vals[1]) < 1e-4
        assert abs(vals[0] - (-83.3235)) < 1e-4
        solns = sig_solrec(dual, ineq_tol=0)
        assert (f(solns[0]) - dual.value) / abs(dual.value) < 0.007
