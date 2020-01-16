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
import numpy as np
import unittest
from sageopt.symbolic.polynomials import Polynomial, standard_poly_monomials, PolyDomain
from sageopt.relaxations.sage_polys import infer_domain
from sageopt import coniclifts as cl


class TestPolynomials(unittest.TestCase):

    @staticmethod
    def are_equal(poly1, poly2):
        diff = poly1 - poly2
        dff = diff.without_zeros()
        return diff.m == 1 and diff.c[0] == 0

    #
    #    Test arithmetic and operator overloading.
    #

    # noinspection PyUnresolvedReferences
    def test_scalar_multiplication(self):
        # data for tests
        alpha0 = np.array([[0], [1], [2]])
        c0 = np.array([1, 2, 3])
        s0 = Polynomial(alpha0, c0)
        # Tests
        s = 2 * s0
        # noinspection PyTypeChecker
        assert set(s.c) == set(2 * s0.c)
        s = s0 * 2
        # noinspection PyTypeChecker
        assert set(s.c) == set(2 * s0.c)
        s = 1 * s0
        assert s.alpha_c == s0.alpha_c
        s = 0 * s0
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}

    def test_addition_and_subtraction(self):
        # data for tests
        s0 = Polynomial(np.array([[0], [1], [2]]),
                        np.array([1, 2, 3]))
        t0 = Polynomial(np.array([[4]]),
                        np.array([5]))
        # tests
        s = s0 - s0
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}
        s = s0 + t0
        assert s.alpha_c == {(0,): 1, (1,): 2, (2,): 3, (4,): 5}

    def test_polynomial_multiplication(self):
        # data for tests
        s0 = Polynomial(np.array([[0], [1], [2]]),
                        np.array([1, 2, 3]))
        t0 = Polynomial(np.array([[1]]),
                        np.array([1]))
        q0 = Polynomial(np.array([[5]]),
                        np.array([0]))
        # tests
        s = s0 * t0
        s = s.without_zeros()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3}
        s = t0 * s0
        s = s.without_zeros()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3}
        s = s0 * q0
        s = s.without_zeros()
        assert s.alpha_c == {(0,): 0}

    def test_polynomial_exponentiation(self):
        p = Polynomial.from_dict({(0,): -1, (1,): 1})
        # square of (x-1)
        res = p ** 2
        expect = Polynomial.from_dict({(0,): 1, (1,): -2, (2,): 1})
        assert res == expect
        # cube of (2x+5)
        p = Polynomial.from_dict({(0,): 5, (1,): 2})
        expect = Polynomial.from_dict({(0,): 125, (1,): 150, (2,): 60, (3,): 8})
        res = p ** 3
        assert res == expect

    def test_composition(self):
        p = Polynomial.from_dict({(2,): 1})  # represents lambda x: x ** 2
        z = Polynomial.from_dict({(1,): 2, (0,): -1})  # represents lambda x: 2*x - 1
        w = p(z)  # represents lambda x: (2*x - 1) ** 2
        assert w(0.5) == 0
        assert w(1) == 1
        assert w(0) == 1
        x = standard_poly_monomials(3)
        p = np.prod(x)
        y = standard_poly_monomials(2)
        expr = np.array([y[0], y[0]-y[1], y[1]])
        w = p(expr)
        assert w.n == 2
        assert w(np.array([1, 1])) == 0
        assert w(np.array([1, -2])) == -6

    def test_standard_monomials(self):
        x = standard_poly_monomials(3)
        y_actual = np.prod(x)
        y_expect = Polynomial.from_dict({(1, 1, 1): 1})
        assert TestPolynomials.are_equal(y_actual, y_expect)
        x = standard_poly_monomials(2)
        y_actual = np.sum(x) ** 2
        y_expect = Polynomial.from_dict({(2, 0): 1, (1, 1): 2, (0, 2): 1})
        assert TestPolynomials.are_equal(y_actual, y_expect)

    def test_polynomial_grad_val(self):
        f = Polynomial.from_dict({(3,): 1, (0,): -1})
        actual = f.grad_val(np.array([0.5]))
        expect = 3*0.5**2
        assert abs(actual[0] - expect) < 1e-8

    def test_polynomial_hess_val(self):
        f = Polynomial.from_dict({(3,): 1, (0,): -1})
        actual = f.hess_val(np.array([0.1234]))
        expect = 3*2*0.1234
        assert abs(actual[0] - expect) < 1e-8

    #
    #   Test construction of [constant] signomial representatives
    #

    def test_sigrep_1(self):
        p = Polynomial.from_dict({(0, 0): -1, (1, 2): 1, (2, 2): 10})
        # One non-even lattice point (the only one) changes sign.
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 0
        assert sr.alpha_c == {(0, 0): -1, (1, 2): -1, (2, 2): 10}

    def test_sigrep_2(self):
        p = Polynomial.from_dict({(0, 0): 0, (1, 1): -1, (3, 3): 5})
        # One non-even lattice point changes sign, another stays the same
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 0
        assert sr.alpha_c == {(0, 0): 0, (1, 1): -1, (3, 3): -5}

    def test_sigrep_3(self):
        alpha = np.random.randint(low=1, high=10, size=(10, 3))
        alpha *= 2
        c = np.random.randn(10)
        p = Polynomial(alpha, c)
        # The signomial representative has the same exponents and coeffs.
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 0
        assert p.alpha_c == sr.alpha_c


class TestPolyDomains(unittest.TestCase):

    def test_infeasible_poly_domain(self):
        x = cl.Variable()
        cons = [x <= -1, x >= 1]
        try:
            dom = PolyDomain(1, logspace_cons=cons)
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'seem to be infeasible' in err_str
        A = np.ones(shape=(2, 2))
        b = np.array([0, 1])
        K = [cl.Cone('0', 2)]
        try:
            dom = PolyDomain(2, log_AbK=(A, b, K))
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'seem to be infeasible' in err_str
        pass

    def test_infer_box_polydomain(self):
        bounds = [(-0.1, 0.4), (0.4, 1),
                  (-0.7, -0.4), (-0.7, 0.4),
                  (0.1, 0.2), (-0.1, 0.2),
                  (-0.3, 1.1), (-1.1, -0.3)]
        x = standard_poly_monomials(8)
        gp_gs = [0.4 ** 2 - x[0] ** 2,
                 1 - x[1] ** 2, x[1] ** 2 - 0.4 ** 2,
                 0.7 ** 2 - x[2] ** 2, x[2] ** 2 - 0.4 ** 2,
                 0.7 ** 2 - x[3] ** 2,
                 0.2 ** 2 - x[4] ** 2, x[4] ** 2 - 0.1 ** 2,
                 0.2 ** 2 - x[5] ** 2,
                 1.1 ** 2 - x[6] ** 2,
                 1.1 ** 2 - x[7] ** 2, x[7] ** 2 - 0.3 ** 2]
        lower_gs = [x[i] - lb for i, (lb, ub) in enumerate(bounds)]
        upper_gs = [ub - x[i] for i, (lb, ub) in enumerate(bounds)]
        gts = lower_gs + upper_gs + gp_gs
        dummy_f = x[0]
        dom = infer_domain(dummy_f, gts, [])
        assert dom.A.shape == (12, 8)
        assert len(dom.gts) == 12
        assert len(dom.eqs) == 0
        x0 = np.array([-0.1, 1, -0.6, 0, 0.2, 0.2, -0.3, -1.05])
        is_in = dom.check_membership(x0, tol=1e-10)
        assert is_in
        x1 = x0.copy()
        x1[7] = -1.11
        is_in = dom.check_membership(x1, tol=1e-5)
        assert not is_in
        pass

    def test_infer_expcone_polydomain(self):
        x = standard_poly_monomials(4)
        g = 1 - np.sum(np.power(x, 2))
        dummy_f = x[0] * 0
        dom = infer_domain(dummy_f, [g], [])
        assert len(dom.K) == 5
        assert dom.A.shape == (13, 8)
        assert dom.K[0].type == '+'
        assert dom.K[0].len == 1
        for i in [1,2,3,4]:
            assert dom.K[i].type == 'e'
        assert dom.b[0] == 1

