import unittest
import numpy as np
import sageopt.coniclifts as cl


class BaseTest(unittest.TestCase):
    # Copied from CVXPY source

    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places: int = 5) -> None:
        if np.isscalar(a):
            a = [a]
        else:
            a = self.mat_to_list(a)
        if np.isscalar(b):
            b = [b]
        else:
            b = self.mat_to_list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 5, delta=None) -> None:
        super(BaseTest, self).assertAlmostEqual(a, b, places=places, delta=delta)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat


class SolverTestHelper:
    # Copied from CVXPY source; requires minimization objective

    def __init__(self, obj_pair, var_pairs, con_pairs) -> None:
        self.objective = obj_pair[0]
        self.constraints = [c for c, _ in con_pairs]
        self.prob = cl.Problem(cl.MIN, self.objective, self.constraints)
        self.variables = [x for x, _ in var_pairs]

        self.expect_val = obj_pair[1]
        self.expect_dual_vars = [dv for _, dv in con_pairs]
        self.expect_prim_vars = [pv for _, pv in var_pairs]
        self.tester = BaseTest()

    def solve(self, solver, **kwargs) -> None:
        self.prob.solve(solver=solver, **kwargs)

    def check_primal_feasibility(self, places) -> None:
        all_cons = [c for c in self.constraints]  # shallow copy
        for con in all_cons:
            viol = con.violation()
            if isinstance(viol, np.ndarray):
                viol = np.linalg.norm(viol, ord=2)
            self.tester.assertAlmostEqual(viol, 0, places)

    def verify_objective(self, places) -> None:
        actual = self.prob.value
        expect = self.expect_val
        if expect is not None:
            self.tester.assertAlmostEqual(actual, expect, places)

    def verify_primal_values(self, places) -> None:
        for idx in range(len(self.variables)):
            actual = self.variables[idx].value
            expect = self.expect_prim_vars[idx]
            if expect is not None:
                self.tester.assertItemsAlmostEqual(actual, expect, places)
