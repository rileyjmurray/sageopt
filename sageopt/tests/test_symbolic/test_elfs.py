"""
   Copyright 2020 Riley John Murray

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
import numpy as np
import unittest
from nose.tools import assert_raises
from scipy.optimize import basinhopping
from sageopt.symbolic.elfs import Elf, spelf
from sageopt.symbolic.signomials import Signomial
from sageopt.coniclifts import Variable, Problem, vector2norm, MIN, MAX, SOLVED
from sageopt.coniclifts import PrimalSageCone


class TestElfs(unittest.TestCase):

    def test_construction(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)  # s(x) = exp(0 * x) - exp(1 * x) - 2 exp(2 * x)
        # Test 1
        f = Elf([s, 0.333])  # f(x) = s(x) + 0.333 * x
        x = np.array([0.75])
        actual = f(x)
        expect = s(x) + 0.333 * x
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # Test 2
        f = Elf([12, s])  # f(x) = 12 + x * s(x)
        actual = f(x)
        expect = 12 + x * s(x)
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)

    def test_addition(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)
        f1 = Elf([s, 0.333])
        f2 = Elf([12, s])
        # test 1: signomial and lenomial
        f3 = s + f1
        x = np.array([1.2345])
        actual = f3(x)
        expect = s(x) + (s(x) + 0.333 * x)
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # test 2: lenomial and lenomial
        f4 = f1 + f2
        actual = f4(x)
        expect = (s(x) + 0.333 * x) + (12 + x * s(x))
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)

    def test_subtraction(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)
        f1 = Elf([s, 0.333])
        f2 = Elf([12, s])
        x = np.array([-0.98765])
        # test 1: lenomial and lenomial (different)
        f3 = f1 - f2
        actual = f3(x)
        expect = (s(x) + 0.333 * x) - (12 + x * s(x))
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # test 2: lenomial and lenomial (same)
        f4 = f1 - f1
        self.assertTrue(f4.is_signomial())
        f4.sig = f4.sig.without_zeros()
        self.assertEqual(1, f4.sig.m)
        self.assertEqual(0, f4.sig.c.item())
        # test 3: lenomial and constant
        f5 = f2 - 12
        self.assertEqual(f2.xsigs[0], f5.xsigs[0])
        self.assertEqual(f5.sig, Signomial.cast(1, 0.0))

    def test_multiplication(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)
        f1 = Elf([s, 0.333])
        f2 = Elf([12, s])
        x = np.array([-0.98765])
        # test 1: lenomial times lenomial (invalid)
        with assert_raises(ArithmeticError):
            f3 = f1 * f2
        # test 2: lenomial times lenomial (valid due to hidden signomial)
        f3 = Elf([s, 0.0])
        f4 = f1 * f3
        actual = f4(x)
        expect = (s(x) + 0.333 * x) * (s(x))
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # test 3: lenomial times explicit signomial
        f5 = s * f2
        actual = f5(x)
        expect = s(x) * (12 + x * s(x))
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # test 4: lenomial times explicit signomial (switch order)
        f6 = f2 * s
        actual = f6(x)
        expect = s(x) * (12 + x * s(x))
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        pass


class TestSpelfs(unittest.TestCase):

    def test_pelf1(self):
        # construct a symbolic positive entropy-like function
        R = np.ones(shape=(1, 3))
        S = np.array([[0.2, 0.6, -0.5]])
        f, pelfcon, melfs = spelf(R, S, zero_origin=False)
        # create a coniclifts problem, which projects a fixed
        #   3-vector onto the set of vectors which are feasible for "f"
        np.random.seed(0)
        c_ref = np.random.randn(3)
        c_ref = c_ref.round(decimals=2)
        c_pelf = pelfcon.variables()[0].ravel()
        t = Variable(name='epi_norm')
        cons = [pelfcon, vector2norm(c_pelf - c_ref) <= t]
        prob = Problem(MIN, t, cons)
        prob.solve(verbose=False)
        # Recover the entropy-like function, use a hueristic method
        #   to check that it's nonnegative.
        g = f.fix_coefficients()
        x0 = np.random.randn(3).round(decimals=2)
        opt_res = basinhopping(g, x0)
        self.assertGreaterEqual(opt_res.fun, -1e-8)

    def _spelf1(self, sage):
        np.random.seed(1)  # seed=0 results in an unbounded Elf.
        # problem data
        alpha = np.random.randn(4).reshape((-1, 1))
        f0 = Signomial(alpha, np.ones(4))
        f1 = Elf.sig_times_linfunc(Signomial.cast(1, 1.0), np.array([1.0]))
        gamma = Variable(name='gamma')
        f = f0 + f1 - gamma
        # constraints
        g, spelf_con, melfs = spelf(f.rmat, f.smat, zero_origin=False)
        cons = [spelf_con]
        if sage:
            h = Signomial(alpha, Variable(shape=(4,)))
            sage_con = PrimalSageCone(h.c, h.alpha, None, 'sage_part')
            cons.append(sage_con)
            delta = f - (g + h)
        else:
            delta = f - g
        sum_con = [delta.sig.c == 0, delta.xsigs[0].c == 0]
        cons += sum_con
        # construct and solve coniclifts Problem
        prob = Problem(MAX, gamma, cons)
        prob.solve(verbose=False)
        self.assertEqual(prob.status, SOLVED)
        self.assertGreater(prob.value, -np.inf)
        # check that the translated function attains a value near zero.
        ffixed = f.fix_coefficients()
        x0 = np.array([1])
        opt_res = basinhopping(ffixed, x0)
        attained_val = opt_res.fun
        self.assertLessEqual(attained_val, 1e-6)

    def test_spelf1_basic(self):
        self._spelf1(sage=False)

    def test_spelf1_sage(self):
        # SAGE doesn't actually help here, since the signomial term
        # of the translated function only has positive coefficients.
        self._spelf1(sage=True)
