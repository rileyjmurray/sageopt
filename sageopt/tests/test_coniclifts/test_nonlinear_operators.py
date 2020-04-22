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
from scipy.special import rel_entr
from sageopt.coniclifts.base import Variable
from sageopt.coniclifts.operators import affine
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.operators.abs import abs as cl_abs
from sageopt.coniclifts.operators.pos import pos as cl_pos
from sageopt.coniclifts.compilers import compile_constrained_system
from sageopt.coniclifts.cones import Cone


class TestNonlinearOperators(unittest.TestCase):

    def test_relent_1(self):
        # compilation and evaluation
        x = Variable(shape=(2,), name='x')
        y = Variable(shape=(2,), name='y')
        re = relent(2 * x, np.exp(1) * y)
        con = [re <= 10,
               3 <= x,
               x <= 5]
        # compilation
        A, b, K, _, _, _ = compile_constrained_system(con)
        A_expect = np.array([[0., 0., 0., 0., -1., -1.],  # linear inequality on epigraph for relent constr
                             [1., 0., 0., 0., 0., 0.],    # bound constraints on x
                             [0., 1., 0., 0., 0., 0.],    #
                             [-1., 0., 0., 0., 0., 0.],   # more bound constraints on x
                             [0., -1., 0., 0., 0., 0.],   #
                             [0., 0., 0., 0., -1., 0.],   # first exponential cone
                             [0., 0., 2.72, 0., 0., 0.],  #
                             [2., 0., 0., 0., 0., 0.],    #
                             [0., 0., 0., 0., 0., -1.],   # second exponential cone
                             [0., 0., 0., 2.72, 0., 0.],  #
                             [0., 2., 0., 0., 0., 0.]])   #
        A = np.round(A.toarray(), decimals=2)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([10., -3., -3., 5., 5., 0., 0., 0., 0., 0., 0.]))
        assert K == [Cone('+', 1), Cone('+', 2), Cone('+', 2), Cone('e', 3), Cone('e', 3)]
        # value propagation
        x0 = np.array([1,2])
        x.value = x0
        y0 = np.array([3,4])
        y.value = y0
        actual = re.value
        expect = np.sum(rel_entr(2 * x0, np.exp(1) * y0))
        assert abs(actual - expect) < 1e-7

    def test_relent_2(self):
        # compilation with the elementwise option
        x = Variable(shape=(2,), name='x')
        y = Variable(shape=(2,), name='y')
        re = affine.sum(relent(2 * x, np.exp(1) * y, elementwise=True))
        con = [re <= 1]
        A, b, K, _, _, _ = compile_constrained_system(con)
        A_expect = np.array([[0., 0., 0., 0., -1., -1.],  # linear inequality on epigraph for relent constr
                             [0., 0., 0., 0., -1., 0.],   # first exponential cone
                             [0., 0., 2.72, 0., 0., 0.],  #
                             [2., 0., 0., 0., 0., 0.],    #
                             [0., 0., 0., 0., 0., -1.],   # second exponential cone
                             [0., 0., 0., 2.72, 0., 0.],  #
                             [0., 2., 0., 0., 0., 0.]])   #
        A = np.round(A.toarray(), decimals=2)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([1., 0., 0., 0., 0., 0., 0.]))
        assert K == [Cone('+', 1), Cone('e', 3), Cone('e', 3)]

    def test_vector2norm_1(self):
        x = Variable(shape=(3,), name='x')
        nrm = vector2norm(x)
        con = [nrm <= 1]
        A_expect = np.array([[0, 0, 0, -1],   # linear inequality constraint in terms of epigraph variable
                             [0, 0, 0,  1],   # epigraph component in second order cone constraint
                             [1, 0, 0,  0],   # start of main block in second order cone constraint
                             [0, 1, 0,  0],
                             [0, 0, 1,  0]])  # end of main block in second order cone constraint
        A, b, K, _1, _2, _3 = compile_constrained_system(con)
        A = np.round(A.toarray(), decimals=1)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([1, 0, 0, 0, 0]))
        assert K == [Cone('+', 1), Cone('S', 4)]
        # value propagation
        x.value = np.zeros(3)
        viol = con[0].violation()
        assert viol == 0

    def test_vector2norm_2(self):
        x = Variable(shape=(3,), name='x')
        y = Variable(shape=(1,), name='y')
        nrm = vector2norm(x - np.array([0.1, 0.2, 0.3]))
        con = [nrm <= y]
        A_expect = np.array([[0, 0, 0, 1, -1],   # linear inequality constraint in terms of epigraph variable
                             [0, 0, 0, 0,  1],   # epigraph component in second order cone constraint
                             [1, 0, 0, 0,  0],   # start of main block in second order cone constraint
                             [0, 1, 0, 0,  0],
                             [0, 0, 1, 0,  0]])  # end of main block in second order cone constraint
        b_expect = np.zeros(shape=(5,))
        b_expect[0] = 0
        b_expect[2] = -0.1
        b_expect[3] = -0.2
        b_expect[4] = -0.3
        A, b, K, _, _, _ = compile_constrained_system(con)
        A = np.round(A.toarray(), decimals=1)
        assert np.all(A == A_expect)
        assert np.all(b == b_expect)
        assert K == [Cone('+', 1), Cone('S', 4)]

    def test_abs_1(self):
        x = Variable(shape=(2,), name='x')
        one_norm = affine.sum(cl_abs(x))
        con = [one_norm <= 5]
        A_expect = np.array([[0,  0, -1, -1],
                             [1,  0, 1,  0],
                             [-1, 0, 1,  0],
                             [0,  1, 0,  1],
                             [0, -1, 0,  1]])
        A, b, K, _1, _2, _3 = compile_constrained_system(con)
        A = np.round(A.toarray(), decimals=1)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([5, 0, 0, 0, 0]))
        assert K == [Cone('+', 1), Cone('+', 2), Cone('+', 2)]
        # value propagation
        x.value = np.array([1, -2])
        viol = con[0].violation()
        assert viol == 0
        x.value = np.array([-3, 3])
        viol = con[0].violation()
        assert viol == 1

    def test_pos_1(self):
        x = Variable(shape=(3,), name='x')
        con = [cl_pos(affine.sum(x)-7) <= 5]
        A_expect = np.array([[0,   0,  0, -1],
                             [0,   0,  0,  1],
                             [-1, -1, -1,  1]])
        A, b, K, _1, _2, _3 = compile_constrained_system(con)
        A = np.round(A.toarray(), decimals=1)
        assert np.all(A == A_expect)
        assert np.all(b == np.array([5, 0, 7]))
        assert K == [Cone('+', 1), Cone('+', 2)]
        # value propagation
        x.value = np.array([4, 4, 4])
        viol = con[0].violation()
        assert viol == 0
        x.value = np.array([4.2, 3.9, 4.0])
        viol = con[0].violation()
        assert abs(viol - 0.1) < 1e-7
