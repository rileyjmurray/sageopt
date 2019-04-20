import numpy as np
import unittest
from sageopt.symbolic.signomials import Signomial


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


if __name__ == '__main__':
    unittest.main()
