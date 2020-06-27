"""
   Copyright 2020 Riley John Murray

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
from sageopt.relaxations.sage_sigs import infer_domain, sig_relaxation, sig_constrained_relaxation
from sageopt.relaxations.sig_solution_recovery import sig_solrec, local_refine
from sageopt.interop.gpkit import GPKIT_INSTALLED
from sageopt.interop.gpkit import gpkit_model_to_sageopt_model


@unittest.skipUnless(GPKIT_INSTALLED, 'These tests are only applicable when GPKit is installed.')
class TestGPKitInterop(unittest.TestCase):

    def test_tiny_sp_1(self):
        """
        A signomial inequality constraint.
        """
        from gpkit import Variable, Model, SignomialsEnabled
        #
        # Build GPKit model
        #
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            constraints = [x >= 1 - y, y <= 0.5]
        gpkm = Model(x, constraints)
        #
        # Recover data for a sageopt model
        #
        som = gpkit_model_to_sageopt_model(gpkm)
        f = som['f']
        gp_ineqs = som['gp_gts']
        X = infer_domain(f, gp_ineqs, [])
        sp_ineqs = som['sp_gts']
        prob = sig_constrained_relaxation(f, sp_ineqs, [], X)
        #
        # Solve the sageopt model and check optimality
        #
        prob.solve(solver='ECOS', verbose=False)
        self.assertAlmostEqual(prob.value, 0.5, places=3)
        soln = sig_solrec(prob)[0]
        soln = local_refine(f, sp_ineqs + gp_ineqs, [], x0=soln)
        self.assertAlmostEqual(f(soln), 0.5, places=3)
        pass

    def test_tiny_sp_2(self):
        """
        A signomial equality constraint
        """
        from gpkit import Variable, Model, SignomialsEnabled
        from gpkit.constraints.sigeq import SingleSignomialEquality
        #
        # Build GPKit model
        #
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            constraints = [0.2 <= x, x <= 0.95, SingleSignomialEquality(x + y, 1)]
        gpkm = Model(x*y, constraints)
        #
        #   Recover data for the sageopt model
        #
        som = gpkit_model_to_sageopt_model(gpkm)
        sp_eqs, gp_gts = som['sp_eqs'], som['gp_gts']
        self.assertEqual(len(sp_eqs), 1)
        self.assertEqual(len(gp_gts), 2)
        self.assertEqual(len(som['sp_gts']), 0)
        self.assertEqual(len(som['gp_eqs']), 0)
        f = som['f']
        X = infer_domain(f, gp_gts, [])
        prob = sig_constrained_relaxation(f, gp_gts, sp_eqs, X,  p=1)
        #
        #   Solve and check solution
        #
        prob.solve(solver='ECOS', verbose=False)
        soln = sig_solrec(prob)[0]
        geo_soln = np.exp(soln)
        vkmap = som['vkmap']
        self.assertAlmostEqual(geo_soln[vkmap[x.key]], 0.95, places=2)
        self.assertAlmostEqual(geo_soln[vkmap[y.key]], 0.05, places=2)

    def test_tiny_gp_1(self):
        """
        min x : x == 1, 2 <= y <= 3
        """
        #
        #   Make the GPKit model
        #
        from gpkit import Variable, Model
        x = Variable('x')
        y = Variable('y')
        constraints = [x == 1, 2 <= y, y <= 3]
        gpkm = Model(x, constraints)
        #
        #   Convert to sageopt model
        #
        som = gpkit_model_to_sageopt_model(gpkm)
        self.assertEqual(len(som['sp_gts']), 0)
        self.assertEqual(len(som['sp_eqs']), 0)
        f = som['f']
        gts = som['gp_gts']
        eqs = som['gp_eqs']
        X = infer_domain(f, gts, eqs)
        prob = sig_relaxation(f, X)
        #
        #   Solve and check solution
        #
        prob.solve(solver='ECOS', verbose=False)
        soln = sig_solrec(prob)[0]
        geo_soln = np.exp(soln)
        vkmap = som['vkmap']
        self.assertAlmostEqual(geo_soln[vkmap[x.key]], 1.0, places=3)
        self.assertGreaterEqual(geo_soln[vkmap[y.key]], 1.999)
        self.assertLessEqual(geo_soln[vkmap[y.key]], 3.001)

    def test_small_gp_1(self):
        """
        Nine variables (6 fixed, 3 free). Inequality constraints only.
        """
        #
        #   Build the GPKit model
        #
        from gpkit import Variable, Model
        # Parameters
        alpha = Variable("alpha", 2, "-", "lower limit, wall aspect ratio")
        beta = Variable("beta", 10, "-", "upper limit, wall aspect ratio")
        gamma = Variable("gamma", 2, "-", "lower limit, floor aspect ratio")
        delta = Variable("delta", 10, "-", "upper limit, floor aspect ratio")
        A_wall = Variable("A_{wall}", 200, "m^2", "upper limit, wall area")
        A_floor = Variable("A_{floor}", 50, "m^2", "upper limit, floor area")
        # Decision variables
        h = Variable("h", "m", "height")
        w = Variable("w", "m", "width")
        d = Variable("d", "m", "depth")
        # Constraints
        constraints = [A_wall >= 2 * h * w + 2 * h * d,
                       A_floor >= w * d,
                       h / w >= alpha,
                       h / w <= beta,
                       d / w >= gamma,
                       d / w <= delta]
        # Objective function
        V = h * w * d
        objective = 1 / V
        # Formulate the Model
        gpkm = Model(objective, constraints)
        gpk_sol = gpkm.solve(verbosity=0)
        #
        #   Infer sageopt model data from the GPKit model
        #
        som = gpkit_model_to_sageopt_model(gpkm)
        f = som['f']
        X = infer_domain(f, som['gp_gts'], som['gp_eqs'])
        prob = sig_relaxation(f, X)
        #
        #   Solve the sageopt model; check optimality.
        #
        prob.solve(solver='ECOS', verbose=False)
        lower = prob.value
        soln = sig_solrec(prob)[0]
        upper = f(soln)
        assert abs(lower - upper) <= 1e-4
        self.assertAlmostEqual(lower, upper, places=4)
        self.assertAlmostEqual(lower, gpk_sol['cost'].m)
        #
        #   Check the sageopt solution against GPKit solution
        #
        geo_soln = np.exp(soln)
        vkmap = som['vkmap']
        so_sol = {vk: geo_soln[vkmap[vk]] for vk in vkmap}
        for vk, gpk_val in gpk_sol['freevariables'].items():
            so_val = so_sol[vk]
            delta = so_val - gpk_val
            self.assertLessEqual(abs(delta), 0.02)
        pass

