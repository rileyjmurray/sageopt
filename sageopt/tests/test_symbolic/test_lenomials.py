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
from sageopt.symbolic.lenomials import Lenomial
from sageopt.symbolic.signomials import Signomial
from sageopt.relaxations import infer_domain
from sageopt import coniclifts as cl


class TestLenomials(unittest.TestCase):

    def test_construction(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)  # s(x) = exp(0 * x) - exp(1 * x) - 2 exp(2 * x)
        # Test 1
        f = Lenomial([s, 0.333])  # f(x) = s(x) + 0.333 * x
        x = np.array([0.75])
        actual = f(x)
        expect = s(x) + 0.333 * x
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)
        # Test 2
        f = Lenomial([12, s])  # f(x) = 12 + x * s(x)
        actual = f(x)
        expect = 12 + x * s(x)
        delta = (actual - expect).item()
        self.assertAlmostEqual(delta, 0.0, places=6)

    def test_addition(self):
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        s = Signomial(alpha, c)
        f1 = Lenomial([s, 0.333])
        f2 = Lenomial([12, s])
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
        f1 = Lenomial([s, 0.333])
        f2 = Lenomial([12, s])
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
        f1 = Lenomial([s, 0.333])
        f2 = Lenomial([12, s])
        x = np.array([-0.98765])
        # test 1: lenomial times lenomial (invalid)
        with assert_raises(ArithmeticError):
            f3 = f1 * f2
        # test 2: lenomial times lenomial (valid due to hidden signomial)
        f3 = Lenomial([s, 0.0])
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





