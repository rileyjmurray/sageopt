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
from sageopt import coniclifts as cl
from sageopt.coniclifts.problems.problem import Problem


class TestSolvers(unittest.TestCase):

    def test_mosek_avoid_slacks(self):
        x = cl.Variable(shape=(2,), name='x')
        y = cl.Variable(shape=(2,), name='y')
        z = cl.Variable(shape=(4,), name='z')
        re = cl.relent(2 * x + 1, np.exp(1) * y)
        t = cl.Variable()
        obj = cl.vector2norm(z)
        con = [re <= 10, 3 <= x, x <= 5, obj <= t, cl.hstack((x, y)) == z]
        prob = Problem(cl.MIN, t, con)
        prob.solve(solver='MOSEK',
                   cache_apply_data=True,
                   compilation_options={'avoid_slacks': True})
        ad0 = prob.solver_apply_data['MOSEK']
        x0 = x.value
        prob.solve(solver='MOSEK',
                   cache_apply_data=True,
                   compilation_options={'avoid_slacks': False})
        ad1 = prob.solver_apply_data['MOSEK']
        x1 = x.value
        assert np.allclose(x0, x1)
        assert ad0[0]['A'].shape[0] < ad1[0]['A'].shape[0]
        pass


if __name__ == '__main__':
    unittest.main()
