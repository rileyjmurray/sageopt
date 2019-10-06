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
from sageopt import coniclifts as cl


class TestPolynomials(unittest.TestCase):

    @staticmethod
    def are_equal(poly1, poly2):
        diff = poly1 - poly2
        diff.remove_terms_with_zero_as_coefficient()
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
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}

    def test_addition_and_subtraction(self):
        # data for tests
        s0 = Polynomial(np.array([[0], [1], [2]]),
                        np.array([1, 2, 3]))
        t0 = Polynomial(np.array([[4]]),
                        np.array([5]))
        # tests
        s = s0 - s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s.remove_terms_with_zero_as_coefficient()
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
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3, (0,): 0}
        s = t0 * s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3, (0,): 0}
        s = s0 * q0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(0,): 0}

    def test_polynomial_exponentiation(self):
        p = Polynomial({(0,): -1, (1,): 1})
        # square of (x-1)
        res = p ** 2 - Polynomial({(0,): 1, (1,): -2, (2,): 1})
        res.remove_terms_with_zero_as_coefficient()
        assert res.m == 1 and set(res.c) == {0}
        # cube of (2x+5)
        p = Polynomial({(0,): 5, (1,): 2})
        res = p ** 3 - Polynomial({(0,): 125, (1,): 150, (2,): 60, (3,): 8})
        res.remove_terms_with_zero_as_coefficient()
        assert res.m == 1 and set(res.c) == {0}

    def test_standard_monomials(self):
        x = standard_poly_monomials(3)
        y_actual = np.prod(x)
        y_expect = Polynomial({(1, 1, 1): 1})
        assert TestPolynomials.are_equal(y_actual, y_expect)
        x = standard_poly_monomials(2)
        y_actual = np.sum(x) ** 2
        y_expect = Polynomial({(2, 0): 1, (1, 1): 2, (0, 2): 1})
        assert TestPolynomials.are_equal(y_actual, y_expect)

    def test_polynomial_grad_val(self):
        f = Polynomial({(3,): 1, (0,): -1})
        actual = f.grad_val(np.array([0.5]))
        expect = 3*0.5**2
        assert abs(actual[0] - expect) < 1e-8

    def test_polynomial_hess_val(self):
        f = Polynomial({(3,): 1, (0,): -1})
        actual = f.hess_val(np.array([0.1234]))
        expect = 3*2*0.1234
        assert abs(actual[0] - expect) < 1e-8

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

    #
    #   Test construction of [constant] signomial representatives
    #

    def test_sigrep_1(self):
        p = Polynomial({(0, 0): -1, (1, 2): 1, (2, 2): 10})
        # One non-even lattice point (the only one) changes sign.
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 0
        assert sr.alpha_c == {(0, 0): -1, (1, 2): -1, (2, 2): 10}

    def test_sigrep_2(self):
        p = Polynomial({(0, 0): 0, (1, 1): -1, (3, 3): 5})
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


if __name__ == '__main__':
    unittest.main()
