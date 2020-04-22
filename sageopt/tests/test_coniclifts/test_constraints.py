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
from sageopt.coniclifts.base import Variable, Expression
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.constraints.set_membership import sage_cones
from sageopt.coniclifts.constraints.set_membership import product_cone, psd_cone
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts import MIN as CL_MIN
from sageopt.symbolic.signomials import SigDomain


class TestConstraints(unittest.TestCase):

    def test_linear_equality_1(self):
        n, m = 3, 2
        np.random.seed(0)
        x0 = np.random.randn(n,).round(decimals=5)
        A = np.random.randn(m, n).round(decimals=5)
        b0 = A @ x0
        x = Variable(shape=(n,), name='x')
        constraint = A @ x == b0

        # Test basic constraint attributes
        assert constraint.epigraph_checked  # equality constraints are automatically checked.
        my_vars = constraint.variables()
        assert len(my_vars) == 1 and my_vars[0].name == x.name

        # Test ability to correctly compute violations
        x.value = x0
        viol = constraint.violation()
        assert viol < 1e-15
        viol = constraint.violation(norm_ord=1)
        assert viol < 1e-15
        x.value = np.zeros(n,)
        viol = constraint.violation()
        assert abs(viol - np.linalg.norm(b0, ord=2)) < 1e-15
        viol = constraint.violation(norm_ord=np.inf)
        assert abs(viol - np.linalg.norm(b0, ord=np.inf)) < 1e-15

    def test_linear_inequality_1(self):
        n, m = 2, 4
        A = np.ones(shape=(m, n))
        x = Variable(shape=(n,), name='x')
        constraint = A @ x >= 0

        # Test basic constraint attributes
        assert not constraint.epigraph_checked
        my_vars = constraint.variables()
        assert len(my_vars) == 1 and my_vars[0].name == x.name

        # Test ability to correctly compute violations
        x0 = np.ones(shape=(n,))
        x.value = x0
        viol = constraint.violation()
        assert viol == 0
        x0 = np.zeros(shape=(n,))
        x0[0] = -1
        x.value = x0
        viol_one_norm = constraint.violation(norm_ord=1)
        assert abs(viol_one_norm - 4) < 1e-15
        viol_inf_norm = constraint.violation(norm_ord=np.inf)
        assert abs(viol_inf_norm - 1) < 1e-15

    #
    #   Ordinary SAGE cones
    #

    def test_ordinary_sage_primal_1(self):
        n, m = 2, 5
        np.random.seed(0)
        alpha = np.random.randn(m, n)
        c = Variable(shape=(m,), name='test_c')
        constraint = sage_cones.PrimalSageCone(c, alpha, X=None, name='test')
        c0 = np.ones(shape=(m,))
        c.value = c0
        viol_default = constraint.violation()
        assert viol_default == 0

    def test_ordinary_sage_primal_2(self):
        n, m = 2, 6
        np.random.seed(0)
        alpha = 1 * np.random.randn(m-1, n)
        conv_comb = np.random.rand(m-1)
        conv_comb /= np.sum(conv_comb)
        alpha_last = alpha.T @ conv_comb
        alpha = np.row_stack([alpha, alpha_last])
        c0 = np.array([1, 2, 3, 4, -0.5, -0.1])
        c = Variable(shape=(m,), name='projected_c0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.PrimalSageCone(c, alpha, X=None, name='test')
        epi_constraint = vector2norm(c - c0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')

        # constraint violations
        v0 = sage_constraint.violation(norm_ord=1, rough=False)
        assert v0 < 1e-6
        v1 = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert v1 < 1e-6

        # certificates
        w4 = sage_constraint.age_witnesses[4].value
        c4 = sage_constraint.age_vectors[4].value
        drop4 = np.array([True, True, True, True, False, True])
        level4 = np.sum(rel_entr(w4[drop4], np.exp(1) * c4[drop4])) - c4[4]
        assert level4 < 1e-6
        w5 = sage_constraint.age_witnesses[5].value
        c5 = sage_constraint.age_vectors[5].value
        drop5 = np.array([True, True, True, True, True, False])
        level5 = np.sum(rel_entr(w5[drop5], np.exp(1) * c5[drop5])) - c5[5]
        assert level5 < 1e-6

    def test_ordinary_sage_dual_1(self):
        # generate a point which has positive distance from the dual SAGE cone
        n, m = 2, 6
        np.random.seed(0)
        alpha = 10 * np.random.randn(m, n)
        v0 = 10 * np.abs(np.random.randn(m)) + 0.01
        v0[0] = -v0[0]
        v = Variable(shape=(m,), name='projected_v0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.DualSageCone(v, alpha, X=None, name='test_con')
        epi_constraint = vector2norm(v - v0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')
        viol = sage_constraint.violation(norm_ord=1, rough=False)
        assert viol < 1e-6
        viol = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert viol < 1e-6
        val = prob.value
        assert val + 1e-6 >= abs(v0[0])

    def test_ordinary_sage_dual_2(self):
        # generate a point with zero distance from the dual SAGE cone
        n, m = 2, 6
        np.random.seed(0)
        alpha = 10 * np.random.randn(m, n)
        x0 = np.random.randn(n) / 10
        v0 = np.exp(alpha @ x0)
        v = Variable(shape=(m,), name='projected_v0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.DualSageCone(v, alpha, X=None, name='test_con')
        epi_constraint = vector2norm(v - v0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')
        viol = sage_constraint.violation(norm_ord=1, rough=False)
        assert viol < 1e-6
        viol = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert viol < 1e-6
        val = prob.value
        assert val < 1e-7

    def test_ordinary_sage_dual_3(self):
        # provide a vector "c" in the dual SAGE cone constructor.
        # generate a point with zero distance from the dual SAGE cone
        n, m = 2, 6
        np.random.seed(0)
        alpha = 10 * np.random.randn(m, n)
        x0 = np.random.randn(n) / 10
        v0 = np.exp(alpha @ x0)
        dummy_vars = Variable(shape=(2,)).scalar_variables()
        c = np.array([1, 2, 3, 4, dummy_vars[0], dummy_vars[1]])
        c = Expression(c)
        v = Variable(shape=(m,), name='projected_v0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.DualSageCone(v, alpha, X=None, name='test_con', c=c)
        epi_constraint = vector2norm(v - v0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')
        viol = sage_constraint.violation(norm_ord=1, rough=False)
        assert viol < 1e-6
        viol = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert viol < 1e-6
        val = prob.value
        assert val < 1e-7

    #
    #   Conditional SAGE cones
    #

    def test_conditional_sage_primal_1(self):
        n, m = 2, 6
        x = Variable(shape=(n,), name='x')
        cons = [1 >= vector2norm(x)]
        gts = [lambda z: 1 - np.linalg.norm(z, 2)]
        eqs = []
        sigdom = SigDomain(n, coniclifts_cons=cons, gts=gts, eqs=eqs)
        np.random.seed(0)
        alpha = 10 * np.random.randn(m, n)
        c0 = np.array([1, 2, 3, 4, -0.5, -0.1])
        c = Variable(shape=(m,), name='projected_c0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.PrimalSageCone(c, alpha, X=sigdom, name='test')
        epi_constraint = vector2norm(c - c0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')
        v0 = sage_constraint.violation(norm_ord=1, rough=False)
        assert v0 < 1e-6
        v1 = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert v1 < 1e-6

    def test_conditional_sage_dual_1(self):
        n, m = 2, 6
        x = Variable(shape=(n,), name='x')
        cons = [1 >= vector2norm(x)]
        gts = [lambda z: 1 - np.linalg.norm(z, 2)]
        eqs = []
        sigdom = SigDomain(n, coniclifts_cons=cons, gts=gts, eqs=eqs)
        np.random.seed(0)
        x0 = np.random.randn(n)
        x0 /= 2 * np.linalg.norm(x0)
        alpha = np.random.randn(m, n)
        c = np.array([1,2,3,4, -0.5, -0.1])
        v0 = np.exp(alpha @ x0)
        v = Variable(shape=(m,), name='projected_v0')
        t = Variable(shape=(1,), name='epigraph_var')
        sage_constraint = sage_cones.DualSageCone(v, alpha, name='test', X=sigdom, c=c)
        epi_constraint = vector2norm(v - v0) <= t
        constraints = [sage_constraint, epi_constraint]
        prob = Problem(CL_MIN, t, constraints)
        prob.solve(solver='ECOS')
        v0 = sage_constraint.violation(norm_ord=1, rough=False)
        assert v0 < 1e-6
        v1 = sage_constraint.violation(norm_ord=np.inf, rough=True)
        assert v1 < 1e-6
        val = prob.value
        assert val < 1e-7



