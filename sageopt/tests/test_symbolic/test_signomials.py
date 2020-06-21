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
from nose.tools import assert_raises
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
        s = Signomial.from_dict(alpha_c)
        recovered_alpha_c = dict()
        for i in range(s.m):
            recovered_alpha_c[tuple(s.alpha[i, :])] = s.c[i]
        assert s.n == 1 and s.m == 3 and alpha_c == recovered_alpha_c

    def test_broadcasting(self):
        # any signomial will do.
        alpha_c = {(0,): 1, (1,): -1, (2,): -2}
        s = Signomial.from_dict(alpha_c)
        other = np.array([1, 2])
        t1 = s + other
        self.assertIsInstance(t1, np.ndarray)
        t2 = other + s
        self.assertIsInstance(t2, np.ndarray)
        delta = t1 - t2
        d1 = delta[0].without_zeros()
        d2 = delta[1].without_zeros()
        self.assertEqual(d1.m, 1)
        self.assertEqual(d2.m, 1)

    def test_exponentiation(self):
        x = standard_sig_monomials(2)
        y0 = (x[0] - x[1])**2
        y1 = x[0]**2 - 2*x[0]*x[1] + x[1]**2
        assert y0 == y1
        z0 = x[0]**0.5
        z1 = Signomial.from_dict({(0.5, 0): 1})
        assert z0 == z1

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
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}

    def test_addition_and_subtraction(self):
        # data for tests
        s0 = Signomial.from_dict({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial.from_dict({(-1,): 5})
        # tests
        s = s0 - s0
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s = s.without_zeros()
        assert s.m == 1 and set(s.c) == {0}
        s = s0 + t0
        assert s.alpha_c == {(-1,): 5, (0,): 1, (1,): 2, (2,): 3}

    def test_invalid_signomial_operations(self):
        x = standard_sig_monomials(2)
        y = standard_sig_monomials(1)
        try:
            s = x[0] + y[0]
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'Cannot add' in err_str
        try:
            s = x[0] * y[0]
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'Cannot multiply' in err_str
        s = sum(x)
        try:
            t = s**0.5
            assert False
        except ValueError as err:
            err_str = str(err)
            assert 'Only signomials with exactly one term' in err_str
        z = cl.Variable()
        s = 0 * s + z  # a Signomial, with only a constant term.
        try:
            t = s ** 2
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'Cannot exponentiate signomials with symbolic coefficients' in err_str
        try:
            y = x[0] ** x[1]
            assert False
        except RuntimeError as err:
            err_str = str(err)
            assert 'Cannot raise a signomial to non-numeric powers.' == err_str
        pass

    def test_signomial_multiplication(self):
        # data for tests
        s0 = Signomial.from_dict({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial.from_dict({(-1,): 1})
        q0 = Signomial.from_dict({(5,): 0})
        # tests
        s = s0 * t0
        s = s.without_zeros()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = t0 * s0
        s = s.without_zeros()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = s0 * q0
        s = s.without_zeros()
        assert s.alpha_c == {(0,): 0}

    def test_signomial_evaluation(self):
        s = Signomial.from_dict({(1,): 1})
        assert s(0) == 1 and abs(s(1) - np.exp(1)) < 1e-10
        zero = np.array([0])
        one = np.array([1])
        assert s(zero) == 1 and abs(s(one) - np.exp(1)) < 1e-10
        zero_one = np.array([[0, 1]])
        assert np.allclose(s(zero_one), np.exp(zero_one), rtol=0, atol=1e-10)

    def test_signomial_grad_val(self):
        f = Signomial.from_dict({(2,): 1, (0,): -1})
        actual = f.grad_val(np.array([3]))
        expect = 2*np.exp(2*3)
        assert abs(actual[0] - expect) < 1e-8

    def test_signomial_hess_val(self):
        f = Signomial.from_dict({(-2,): 1, (0,): -1})
        actual = f.hess_val(np.array([3]))
        expect = 4*np.exp(-2*3)
        assert abs(actual[0] - expect) < 1e-8


class TestSigDomain(unittest.TestCase):

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
        for di in [d0, d1, d2]:
            assert len(di.K) == 4
            assert di.K[0].type == '+'
            assert di.K[0].len == 1
            for j in [1, 2, 3]:
                assert di.K[j].type == 'e'
        for di in [d0, d1, d2]:
            A = di.A
            assert np.allclose(A[0, :], np.array([0, -1, -1, -1]))
            selector = np.zeros(shape=(A.shape[0],), dtype=bool)
            selector[[1, 4, 7]] = True
            coeffs = np.sort(A[selector, 0])
            assert np.allclose(coeffs, np.array([-1, 1, 2]))
            assert np.allclose(A[~selector, 0], np.zeros(A.shape[0] - 3))
            compare = np.zeros(shape=(A.shape[0]-1, A.shape[1]-1))
            compare[1, 0] = 1
            compare[4, 1] = 1
            compare[7, 2] = 1
            assert np.allclose(A[1:, 1:], compare)
        pass

    def test_freecomponent_infer_sigdomain(self):
        x = standard_sig_monomials(4)
        dummy_f = x[0] * 0
        gts = [1-x[1]**0.5-x[2]**3]
        dom = infer_domain(dummy_f, gts, [])
        A, b, K = dom.A, dom.b, dom.K
        # ^ Two exponential cones, two epigraph variables, one linear inequality
        assert A.shape == (7, 6)
        assert len(K) == 3
        assert K[0].type == '+' and K[0].len == 1
        assert K[1].type == 'e' and K[1].len == 3
        assert K[2].type == 'e' and K[2].len == 3
        assert np.count_nonzero(A[:, 0]) == 0
        assert np.count_nonzero(A[:, 3]) == 0
        pass
