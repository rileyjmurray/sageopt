"""
   Copyright 2019 Riley John Murray

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
from sageopt.symbolic.polynomials import Polynomial, standard_poly_monomials
from sageopt.relaxations import poly_relaxation, poly_constrained_relaxation, sage_multiplier_search
from sageopt.relaxations import poly_solution_recovery, infer_domain
from sageopt import coniclifts as cl


def primal_dual_unconstrained(p, poly_ell, sigrep_ell, X=None, solver='ECOS'):
    prim = poly_relaxation(p, form='primal', X=X,
                           poly_ell=poly_ell, sigrep_ell=sigrep_ell)
    res1 = prim.solve(solver=solver, verbose=False)
    dual = poly_relaxation(p, form='dual', X=X,
                           poly_ell=poly_ell, sigrep_ell=sigrep_ell)
    res2 = dual.solve(solver=solver, verbose=False)
    return [res1[1], res2[1]]


def primal_dual_constrained(f, gt, eq, p, q, ell, X=None, solver='ECOS'):
    prim = poly_constrained_relaxation(f, gt, eq, form='primal',
                                       p=p, q=q, ell=ell, X=X)
    res1 = prim.solve(solver=solver, verbose=False)
    dual = poly_constrained_relaxation(f, gt, eq, form='dual',
                                       p=p, q=q, ell=ell, X=X)
    res2 = dual.solve(solver=solver, verbose=False)
    return [res1[1], res2[1]], dual


class TestSagePolynomials(unittest.TestCase):

    #
    #   Test non-constant signomial representatives
    #

    def test_sigrep_1(self):
        p = Polynomial.from_dict({(0, 0): -1, (1, 2): 1, (2, 2): 10})
        gamma = cl.Variable(shape=(), name='gamma')
        p -= gamma
        sr, sr_cons = p.sig_rep
        # Even though there is a Variable in p.c, no auxiliary
        # variables should have been introduced by defining this
        # signomial representative.
        assert len(sr_cons) == 0
        count_nonconstants = 0
        for i, ci in enumerate(sr.c):
            if isinstance(ci, cl.base.ScalarExpression):
                if not ci.is_constant():
                    assert len(ci.variables()) == 1
                    count_nonconstants += 1
                    assert ci.variables()[0].name == 'gamma'
            elif sr.alpha[i, 0] == 1 and sr.alpha[i, 1] == 2:
                assert ci == -1
            elif sr.alpha[i, 0] == 2 and sr.alpha[i, 1] == 2:
                assert ci == 10
            else:
                assert False
        assert count_nonconstants == 1

    def test_sigrep_2(self):
        c33 = cl.Variable(shape=(), name='c33')
        alpha = np.array([[0, 0], [1, 1], [3, 3]])
        c = cl.Expression([0, -1, c33])
        p = Polynomial(alpha, c)
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 2
        var_names = set(v.name for v in sr_cons[0].variables())
        var_names.union(set(v.name for v in sr_cons[1].variables()))
        for v in var_names:
            assert v == 'c33' or v == str(p) + ' variable sigrep coefficients'
        assert sr.alpha_c[(1, 1)] == -1

    #
    #   Test unconstrained relaxations
    #

    def test_unconstrained_1(self):
        # Background
        #
        #       This example was constructed soley as a test case for sageopt.
        #
        #       We consider two polynomial optimization problems, that are related
        #       to one another by a change of sign in one of the variables.
        #
        # Tests
        #
        #       (1) Verify primal / dual consistency for the two problems, at level (0, 0).
        #
        #       (2) Verify that the SAGE bound is the same for the two formulations.
        #
        alpha = np.array([[0, 0], [1, 1], [2, 2], [0, 2], [2, 0]])
        # First formulation
        p = Polynomial(alpha, np.array([1, -3, 1, 4, 4]))
        res0 = primal_dual_unconstrained(p, 0, sigrep_ell=0)
        assert abs(res0[0] - res0[1]) <= 1e-6
        # Second formulation
        p = Polynomial(alpha, np.array([1, 3, 1, 4, 4]))
        res1 = primal_dual_unconstrained(p, 0, sigrep_ell=0)
        assert abs(res1[0] - res1[1]) <= 1e-6
        # Check for same results between the two formulations
        expected = 1
        assert abs(res0[0] - expected) <= 1e-5
        assert abs(res1[0] - expected) <= 1e-5

    def test_unconstrained_2(self):
        # Background
        #
        #       Unconstrained minimization of a polynomial in 2 variables.
        #       This is Example 4.1 from a 2018 paper by Seidler and de Wolff
        #       (https://arxiv.org/abs/1808.08431).
        #
        # Tests
        #
        #       (1) primal / dual consistency for (poly_ell, sigrep_ell) \in {(0, 0), (1, 0), (0, 1)}.
        #
        #       (2) Show that the bound with (poly_ell=0, sigrep_ell=1) is strong than
        #           the bound with (poly_ell=1, sigrep_ell=0).
        #
        # Notes
        #
        #       The global minimum of this polynomial (as verified by gloptipoly3) is 0.85018.
        #
        #       The furthest we could progress up the hierarchy before encountering a solver failure
        #       was (poly_ell=0, sigrep_ell=5). In this case the SAGE bound was 0.8336.
        #
        p = Polynomial.from_dict({(0, 0): 1,
                        (2, 6): 3,
                        (6, 2): 2,
                        (2, 2): 6,
                        (1, 2): -1,
                        (2, 1): 2,
                        (3, 3): -3})
        res00 = primal_dual_unconstrained(p, poly_ell=0, sigrep_ell=0)
        expect00 = 0.6932
        assert abs(res00[0] - res00[1]) <= 1e-6
        assert abs(res00[0] - expect00) <= 1e-3
        res10 = primal_dual_unconstrained(p, poly_ell=1, sigrep_ell=0)
        expect10 = 0.7587
        assert abs(res10[0] - res10[1]) <= 1e-5
        assert abs(res10[0] - expect10) <= 1e-3
        if cl.Mosek.is_installed():
            # ECOS fails
            res01 = primal_dual_unconstrained(p, poly_ell=0, sigrep_ell=1, solver='MOSEK')
            expect01 = 0.7876
            assert abs(res01[0] - res01[1]) <= 1e-5
            assert abs(res01[0] - expect01) <= 1e-3

    def test_unconstrained_3(self):
        # Minimization of the six-hump camel back function.
        p = Polynomial.from_dict({(0, 0): 0,
                        (2, 0): 4,
                        (1, 1): 1,
                        (0, 2): -4,
                        (4, 0): -2.1,
                        (0, 4): 4,
                        (6, 0): 1.0 / 3.0})
        # sigrep_ell=0 has a decent bound, and sigrep_ell=1 is nearly optimal.
        # ECOS is unable to solver sigrep_ell=2 due to conditioning problems.
        # MOSEK easily solves sigrep_ell=2, and this is globally optimal
        res00 = primal_dual_unconstrained(p, poly_ell=0, sigrep_ell=0)
        expect00 = -1.18865
        assert abs(res00[0] - res00[1]) <= 1e-6
        assert abs(res00[0] - expect00) <= 1e-3
        res10 = primal_dual_unconstrained(p, poly_ell=1, sigrep_ell=0)
        expect10 = -1.03416
        assert abs(res10[0] - res10[1]) < 1e-6
        assert abs(res10[0] - expect10) <= 1e-3
        if cl.Mosek.is_installed():
            res01 = primal_dual_unconstrained(p, poly_ell=0, sigrep_ell=1, solver='MOSEK')
            expect01 = -1.03221
            assert abs(res01[0] - res01[1]) <= 1e-6
            assert abs(res01[0] - expect01) <= 1e-3
            res02 = primal_dual_unconstrained(p, poly_ell=0, sigrep_ell=2, solver='MOSEK')
            expect02 = -1.0316
            assert abs(res02[0] - res02[1]) < 1e-6
            assert abs(res02[0] - expect02) <= 1e-3

    #
    #   Test constrained relaxations
    #

    def test_ordinary_constrained_1(self):
        # Background
        #
        #       This polynomial is "wrig_5" from a 2008 paper by Ray and Nataraj.
        #       We minimize and maximize this polynomial over the box [-5, 5]^5 \subset R^5.
        #       The minimum and maximum are reported as -30.25 and 40, respectively.
        #       The reported bounds can be certified by SAGE relaxations.
        #
        # Tests
        #
        #       (1) minimization : similar primal / dual objectives for (p, q, ell) = (0, 1, 0).
        #
        #       (2) maximization : similar primal / dual objectives for (p, q, ell) = (0, 2, 0).
        n = 5
        x = standard_poly_monomials(n)
        f = x[4] ** 2 + x[0] + x[1] + x[2] + x[3] - x[4] - 10
        lower_gs = [x[i] - (-5) for i in range(n)]
        upper_gs = [5 - x[i] for i in range(n)]
        gts = lower_gs + upper_gs
        claimed_min = -30.25
        claimed_max = 40
        res_min, _ = primal_dual_constrained(f, gts, [], 0, 1, 0, None)
        assert abs(res_min[0] - claimed_min) < 1e-5
        assert abs(res_min[1] - claimed_min) < 1e-5
        res_max, _ = primal_dual_constrained(-f, gts, [], 0, 2, 0, None)
        res_max = [-res_max[0], -res_max[1]]
        assert abs(res_max[0] - claimed_max) < 1e-5
        assert abs(res_max[1] - claimed_max) < 1e-5

    def test_ordinary_constrained_2(self):
        x = standard_poly_monomials(1)[0]
        f = -x**2
        gts = [1 - x, x - (-1)]
        eqs = []
        res, dual = primal_dual_constrained(f, gts, eqs, 0, 2, 0, None)
        expect = -1
        assert abs(res[0] - expect) < 1e-6
        assert abs(res[1] - expect) < 1e-6
        sols = poly_solution_recovery.poly_solrec(dual, ineq_tol=0, eq_tol=0)
        assert len(sols) > 0
        x0 = sols[0]
        assert f(x0) >= expect

    #
    #   Test multiplier search
    #

    def test_multiplier_search(self):
        # Background
        #
        #       This example comes from Proposition 14 of a 2017 paper by Ahmadi and Majumdar.
        #       It concerns nonnegativity of the polynomial
        #           p(x1, x2, x3) = (x1 + x2 + x3)**2 + a*(x1**2 + x2**2 + x3**2)
        #       for values of "a" in (0, 1).
        #
        # Tests
        #
        #       (1) Find a SAGE polynomial "f1" (over the same exponents as p) so that
        #           when a = 0.3, the product f1 * p is a SAGE polynomial.
        #
        #       (2) Find a SAGE polynomial "f2" (over the same exponents as p**2) so that
        #           when a = 0.15, the product f2 * p is a SAGE polynomial.
        #
        # Notes
        #
        #       In a previous version of sageopt, ECOS could also be run on these tests, but
        #       we needed larger values of "a" to avoid solver failures. Now, ECOS cannot
        #       solve a level 2 relaxation for any interesting value of "a". It is not known at
        #       what point sageopt's problem compilation started generating problems that ECOS
        #       could not solve.
        #
        x = standard_poly_monomials(3)
        p = (np.sum(x)) ** 2 + 0.35 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        res1 = sage_multiplier_search(p, level=1).solve(verbose=False)
        assert abs(res1[1]) < 1e-8
        if cl.Mosek.is_installed():
            p -= 0.2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            res2 = sage_multiplier_search(p, level=2).solve(verbose=False)
            assert abs(res2[1]) < 1e-8

    #
    #   Test conditional SAGE relaxations
    #

    def test_conditional_sage_1(self):
        x = standard_poly_monomials(1)[0]
        f = -x**2
        gts = [1 - x**2]
        opt = -1
        X = infer_domain(f, gts, [])
        res_uncon00 = primal_dual_unconstrained(f, 0, 0, X)
        assert abs(res_uncon00[0] - opt) < 1e-6
        assert abs(res_uncon00[1] - opt) < 1e-6
        res_con010, dual = primal_dual_constrained(f, [], [], 0, 1, 0, X)
        assert abs(res_con010[0] - opt) < 1e-6
        assert abs(res_con010[1] - opt) < 1e-6
        solns = poly_solution_recovery.poly_solrec(dual)
        x_star = solns[0]
        gap = abs(f(x_star) - opt)
        assert gap < 1e-6

    def test_conditional_sage_2(self):
        n = 5
        x = standard_poly_monomials(n)
        f = 0
        for i in range(n):
            sel = np.ones(n, dtype=bool)
            sel[i] = False
            f -= 16 * np.prod(x[sel])
        gts = [0.25 - x[i] ** 2 for i in range(n)]  # -0.5 <= x[i] <= 0.5 for all i.
        opt = -5
        X = infer_domain(f, gts, [])
        res_uncon00 = primal_dual_unconstrained(f, 0, 0, X)
        assert abs(res_uncon00[0] - opt) < 1e-6
        assert abs(res_uncon00[1] - opt) < 1e-6
        res_con010, dual = primal_dual_constrained(f, [], [], 0, 1, 0, X)
        assert abs(res_con010[0] - opt) < 1e-4
        assert abs(res_con010[1] - opt) < 1e-4
        solns = poly_solution_recovery.poly_solrec(dual)
        x_star = solns[0]
        gap = abs(f(x_star) - opt)
        assert gap < 1e-6

    def test_conditional_sage_3(self):
        n = 5
        x = standard_poly_monomials(n)
        f = 0
        for i in range(n):
            sel = np.ones(n, dtype=bool)
            sel[i] = False
            f += 2**(n-1) * np.prod(x[sel])
        gts = [0.25 - x[i] ** 2 for i in range(n)]  # -0.5 <= x[i] <= 0.5 for all i.
        opt = -3
        expect = -5
        X = infer_domain(f, gts, [])
        res_con010, dual = primal_dual_constrained(f, [], [], 0, 1, 0, X)
        assert abs(res_con010[0] - expect) < 1e-4
        assert abs(res_con010[1] - expect) < 1e-4
        solns = poly_solution_recovery.poly_solrec(dual)
        assert len(solns) > 0
        x_star = solns[0]
        gap = abs(f(x_star) - opt)
        assert gap < 1e-4
        pass

    @unittest.skipUnless(cl.Mosek.is_installed(), 'ECOS fails on this problem')
    def test_conditional_sage_4(self):
        n = 4
        x = standard_poly_monomials(n)
        f0 = -x[0]*x[2]**3 + 4*x[1]*x[2]**2*x[3] + 4*x[0]*x[2]*x[3]**2
        f1 = 2*x[1]*x[3]**3 + 4*x[0]*x[2] + 4*x[2]**2 - 10*x[1]*x[3] - 10*x[3]**2 + 2
        f = f0 + f1
        sign_sym = [0.25 - x[i]**2 for i in range(n)]
        X = infer_domain(f, sign_sym, [])
        gts = [x[i] + 0.5 for i in range(n)] + [0.5 - x[i] for i in range(n)]
        dual = poly_constrained_relaxation(f, gts, [], X, p=1, q=2)
        dual.solve()
        expect = -3.180096
        self.assertAlmostEqual(dual.value, expect, places=5)
        solns = poly_solution_recovery.poly_solrec(dual)
        self.assertGreaterEqual(len(solns), 1)
        x_star = solns[0]
        gap = f(x_star) - dual.value
        self.assertLessEqual(gap, 1e-5)
