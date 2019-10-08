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
from sageopt.symbolic.signomials import Signomial, SigDomain, standard_sig_monomials
from sageopt.relaxations import infer_domain
from sageopt import coniclifts as cl


class TestSignomials(unittest.TestCase):

    def test_construction(self):
        # data for tests
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        alpha_c = {(0,): 1, (1,): -1, (2,): -2}
        # Construction with two numpy arrays as arguments
        s = Signomial(alpha, c)
        assert s.n == 1 and s.m == 3 and s.alpha_c == alpha_c
        # Construction with a vector-to-coefficient dictionary
        s = Signomial(alpha_c)
        recovered_alpha_c = dict()
        for i in range(s.m):
            recovered_alpha_c[tuple(s.alpha[i, :])] = s.c[i]
        assert s.n == 1 and s.m == 3 and alpha_c == recovered_alpha_c

    # noinspection PyUnresolvedReferences
    def test_scalar_multiplication(self):
        # data for tests
        alpha0 = np.array([[0], [1], [2]])
        c0 = np.array([1, 2, 3])
        s0 = Signomial(alpha0, c0)
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
        s0 = Signomial({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial({(-1,): 5})
        # tests
        s = s0 - s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = s0 + t0
        assert s.alpha_c == {(-1,): 5, (0,): 1, (1,): 2, (2,): 3}

    def test_signomial_multiplication(self):
        # data for tests
        s0 = Signomial({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial({(-1,): 1})
        q0 = Signomial({(5,): 0})
        # tests
        s = s0 * t0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = t0 * s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = s0 * q0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(0,): 0}

    def test_signomial_evaluation(self):
        s = Signomial({(1,): 1})
        assert s(0) == 1 and abs(s(1) - np.exp(1)) < 1e-10
        zero = np.array([0])
        one = np.array([1])
        assert s(zero) == 1 and abs(s(one) - np.exp(1)) < 1e-10
        zero_one = np.array([[0, 1]])
        assert np.allclose(s(zero_one), np.exp(zero_one), rtol=0, atol=1e-10)

    def test_signomial_grad_val(self):
        f = Signomial({(2,): 1, (0,): -1})
        actual = f.grad_val(np.array([3]))
        expect = 2*np.exp(2*3)
        assert abs(actual[0] - expect) < 1e-8

    def test_signomial_hess_val(self):
        f = Signomial({(-2,): 1, (0,): -1})
        actual = f.hess_val(np.array([3]))
        expect = 4*np.exp(-2*3)
        assert abs(actual[0] - expect) < 1e-8

    def test_infeasible_sig_domain(self):
        x = cl.Variable()
        cons = [x <= -1, x >= 1]
        try:
            dom = SigDomain(1, coniclifts_cons=cons)
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'seem to be infeasible' in err_str
        A = np.ones(shape=(2, 2))
        b = np.array([0, 1])
        K = [cl.Cone('0', 2)]
        try:
            dom = SigDomain(2, AbK=(A, b, K))
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'seem to be infeasible' in err_str
        pass

    def test_eqcon_infer_sigdomain(self):
        y = standard_sig_monomials(3)
        expx0 = np.exp([1, 2, 3])
        eqs = [y[0] - expx0[0], 2*y[1] - 2*expx0[1], 0.5*expx0[2] - 0.5*y[2]]
        gts = [y[0] - y[1] + y[2]]  # not convex
        f = -np.sum(y) + 1
        X = infer_domain(f, gts, eqs)
        is_in = X.check_membership(np.array([1, 2, 3]), tol=1e-10)
        assert is_in
        is_in = X.check_membership(np.array([1.0001, 2, 3]), tol=1e-5)
        assert not is_in
        assert X.A.shape == (3, 3)
        assert all([co.type == '0' for co in X.K])
        residual = X.A @ np.array([1, 2, 3]) + X.b
        assert np.allclose(residual, np.zeros(3))

    def test_expcone_infer_sigdomain(self):
        y = standard_sig_monomials(1)[0]
        dummy_f = y * 0
        g0 = 10 - y - y**2 - 1/y
        d0 = infer_domain(dummy_f, [g0], [])
        g1 = y * g0
        d1 = infer_domain(dummy_f, [g1], [])
        g2 = g0 / y**0.5
        d2 = infer_domain(dummy_f, [g2], [])
        assert np.allclose(d0.A, d1.A)
        assert np.allclose(d0.A, d2.A)
        assert np.allclose(d0.b, d1.b)
        assert np.allclose(d0.b, d2.b)
        for di in [d0, d1, d2]:
            assert len(di.K) == 4
            assert di.K[0].type == '+'
            assert di.K[0].len == 1
            for j in [1,2,3]:
                assert di.K[j].type == 'e'
        pass

if __name__ == '__main__':
    unittest.main()
