import unittest
import numpy as np
from sageopt.coniclifts.base import Variable
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.compilers import compile_constrained_system
from sageopt.coniclifts.cones import Cone


class TestOperators(unittest.TestCase):

    def test_relent(self):
        x = Variable(shape=(2,), name='x')
        y = Variable(shape=(2,), name='y')
        re = relent(2 * x, np.exp(1) * y)
        con = [re <= 10,
               3 <= x,
               x <= 5]
        A, b, K, _, _ = compile_constrained_system(con)
        A_expect = np.array([[0., 0., 0., 0., -1., -1.],
                             [1., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0.],
                             [-1., 0., 0., 0., 0., 0.],
                             [0., -1., 0., 0., 0., 0.],
                             [0., 0., 0., 0., -1., 0.],
                             [0., 0., 2.72, 0., 0., 0.],
                             [2., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., -1.],
                             [0., 0., 0., 2.72, 0., 0.],
                             [0., 2., 0., 0., 0., 0.]])
        A = np.round(A.toarray(), decimals=2)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([10., -3., -3., 5., 5., 0., 0., 0., 0., 0., 0.]))
        assert K == [Cone('+', 1), Cone('+', 2), Cone('+', 2), Cone('e', 3), Cone('e', 3)]


if __name__ == '__main__':
    unittest.main()
