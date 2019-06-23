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
from sageopt.coniclifts.base import Variable
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.constraints.set_membership import sage_cone, conditional_sage_cone, product_cone, psd_cone


class TestConstraints(unittest.TestCase):

    def test_elementwise_equality_1(self):
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
        x.set_scalar_variables(x0)
        viol = constraint.violation()
        assert viol < 1e-15
        viol = constraint.violation(norm_ord=1)
        assert viol < 1e-15
        x.set_scalar_variables(np.zeros(n,))
        viol = constraint.violation()
        assert abs(viol - np.linalg.norm(b0, ord=2)) < 1e-15
        viol = constraint.violation(norm_ord=np.inf)
        assert abs(viol - np.linalg.norm(b0, ord=np.inf)) < 1e-15

    def test_elementwise_inequality_1(self):
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
        x.set_scalar_variables(x0)
        viol = constraint.violation()
        assert viol == 0
        x0 = np.zeros(shape=(n,))
        x0[0] = -1
        x.set_scalar_variables(x0)
        viol_one_norm = constraint.violation(norm_ord=1)
        assert abs(viol_one_norm - 4) < 1e-15
        viol_inf_norm = constraint.violation(norm_ord=np.inf)
        assert abs(viol_inf_norm - 1) < 1e-15

    def test_ordinary_sage_primal_1(self):
        n, m = 2, 5
        np.random.seed(0)
        alpha = np.random.randn(m, n)
        c = Variable(shape=(m,), name='test_c')
        constraint = sage_cone.PrimalSageCone(c, alpha, name='test')
        c0 = np.ones(shape=(m,))
        c.set_scalar_variables(c0)
        viol_default = constraint.violation()
        assert viol_default == 0

