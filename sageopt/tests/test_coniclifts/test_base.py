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
from sageopt.coniclifts.operators import affine
from sageopt.coniclifts.base import *
from sageopt.coniclifts.compilers import find_variables_from_constraints


def verify_entries(expected, actual, tol=1e-6):
    for tup in array_index_iterator(actual.shape):
        temp_exp = expected[tup]
        if len(actual[tup].atoms_to_coeffs) != len(temp_exp.atoms_to_coeffs):
            return False
        else:
            for k, v in actual[tup].atoms_to_coeffs.items():
                if abs(v - temp_exp.atoms_to_coeffs[k]) > tol:
                    print('tup: ' + str(tup))
                    print('first arg  :' + str((k, v)))
                    print('second arg :' + str((k, temp_exp.atoms_to_coeffs[k])))
                    return False
        if not actual[tup].offset == expected[tup].offset:
            print('first  arg: ' + str(expected[tup].offset))
            print('second arg: ' + str(actual[tup].offset))
            return False
        else:
            return True


class TestBase(unittest.TestCase):

    def test_factorization(self):
        n = 3
        x = Variable(shape=(n, n))
        (A, X, B) = x.factor()
        X = Expression(X)
        expr = np.dot(A, X)

        assert verify_entries(expr, x)

        T1 = np.random.randn(6, 3).round(decimals=4)
        expr1 = affine.tensordot(T1, x)
        (A1, X1, B1) = expr1.factor()
        X1 = Expression(X1)
        assert verify_entries(affine.tensordot(A1, X1) + B1, expr1)

        T2 = np.random.randn(2, 6).round(decimals=4)
        expr2 = affine.tensordot(T2, expr1)
        (A2, X2, B2) = expr2.factor()
        X2 = Expression(X2)
        assert verify_entries(affine.tensordot(A2, X2) + B2, expr2)

        Tall = affine.tensordot(T2, A1)
        res3 = affine.tensordot(Tall, X1)
        assert verify_entries(res3, expr2)

    # noinspection PyUnusedLocal
    def test_LMI_input_validation(self):
        y = Variable(shape=(2, 2), name='y')
        msg = ''
        try:
            constr = y >> 0
        except RuntimeError as e:
            msg = str(e)
        assert msg == 'Argument to LMI was not symmetric.'
        z = (y + y.T) * 0.5
        constr = z >> 0
        assert True

    def test_variable_tracking(self):
        x = Variable((5,), name='x')
        y = Variable((3, 4), name='y')

        # Should have a single integer.
        val1 = np.unique([id(v.parent) for v in x.scalar_variables()])
        assert val1.size == 1
        # Should have a single integer (distinct from above).
        val2 = np.unique([id(v.parent) for v in y.scalar_variables()])
        assert val2.size == 1 and val2[0] != val1[0]
        # Should have single integer (same as immediately above).
        val3 = np.unique([id(v.parent) for v in y[:5].scalar_variables()])
        assert val3.size == 1 and val3[0] == val2[0]
        assert x.is_proper()
        assert y.is_proper()
        assert not y[:3].is_proper()

        # This variable does nothing for the first test.
        z = Variable((2, 2, 2), name='z')

        # First statement should be "True", second should be ['x','y']
        constrs1 = [affine.sum(x) == 1, affine.sum(x) - affine.sum(y) <= 0]
        vars1 = find_variables_from_constraints(constrs1)
        assert all(v.is_proper() for v in vars1)
        assert [v.name for v in vars1] == ['x', 'y']

        # First statement should be "False", second should be "True", third should be ['y', 'z']
        zz = z[:, :, 1]
        assert not zz.is_proper()
        constrs2 = [y == 0, affine.trace(zz) >= 10]
        vars2 = find_variables_from_constraints(constrs2)
        assert all(v.is_proper() for v in vars2)
        assert [v.name for v in vars2] == ['y', 'z']

    def test_matmul_1(self):
        x = Variable(shape=(4,))
        c = np.array([1, 2, 3, 4])
        y = c @ x
        assert y.shape == tuple()


if __name__ == '__main__':
    unittest.main()
