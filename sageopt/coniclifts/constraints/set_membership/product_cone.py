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
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.base import Expression, ScalarVariable, Variable
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.problems.problem import Problem


class PrimalProductCone(SetMembership):

    _CONSTRAINT_ID_ = 0

    def __init__(self, y, K):
        self.id = PrimalProductCone._CONSTRAINT_ID_
        PrimalProductCone._CONSTRAINT_ID_ += 1
        if not isinstance(y, Expression):
            y = Expression(y)
        K_size = sum([co.len for co in K])
        if not y.size == K_size:
            raise RuntimeError('Incompatible dimensions for y (' + str(y.size) + ') and K (' + str(K_size) + ')')
        if np.any([co.type == 'P' for co in K]):
            raise RuntimeError('This function does not currently support the PSD cone.')
        self.y = y.ravel()
        self.K = K
        self.K_size = K_size
        pass

    def variables(self):
        return self.y.variables()

    def conic_form(self):
        A_rows, A_cols, A_vals = [], [], []
        b = np.zeros(shape=(self.K_size,))
        for i, se in enumerate(self.y.flat):
            if len(se.atoms_to_coeffs) == 0:
                b[i] = se.offset
                A_rows.append(i)
                A_cols.append(ScalarVariable.curr_variable_count() - 1)
                A_vals.append(0)  # make sure scipy infers correct dimensions later on.
            else:
                b[i] = se.offset
                A_rows += [i] * len(se.atoms_to_coeffs)
                col_idx_to_coeff = [(a.id, c) for a, c in se.atoms_to_coeffs.items()]
                A_cols += [atom_id for (atom_id, _) in col_idx_to_coeff]
                A_vals += [c for (_, c) in col_idx_to_coeff]
        return [(A_vals, np.array(A_rows), A_cols, b, self.K)]

    @staticmethod
    def project(item, K):
        from sageopt.coniclifts import MIN as CL_MIN
        item = Expression(item).ravel()
        x = Variable(shape=(item.size,))
        t = Variable(shape=(1,))
        cons = [vector2norm(item - x) <= t, PrimalProductCone(x, K)]
        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)
        return prob.value

    def __contains__(self, item, tol=1e-7):
        residual = PrimalProductCone.project(item, self.K)
        return residual < tol


class DualProductCone(SetMembership):

    _CONSTRAINT_ID_ = 0

    def __init__(self, y, K):
        # y must belong to K^dagger
        self.id = DualProductCone._CONSTRAINT_ID_
        DualProductCone._CONSTRAINT_ID_ += 1
        if not isinstance(y, Expression):
            y = Expression(y)
        K_size = sum([co.len for co in K])
        if not y.size == K_size:
            raise RuntimeError('Incompatible dimensions for y (' + str(y.size) + ') and K (' + str(K_size) + ')')
        if np.any([co.type == 'P' for co in K]):
            raise RuntimeError('This function does not currently support the PSD cone.')
        self.y = y.ravel()
        self.K = K
        self.K_size = K_size
        pass

    def variables(self):
        return self.y.variables()

    def conic_form(self):
        self_dual_cones = {'+', 'S', 'P'}
        start_row = 0
        y_mod = []
        for co in self.K:
            stop_row = start_row + co.len
            if co.type in self_dual_cones:
                y_mod.append(self.y[start_row:stop_row])
            elif co.type == 'e':
                temp_y = np.array([-self.y[start_row + 2],
                                   np.exp(1) * self.y[start_row + 1],
                                   -self.y[start_row]])
                y_mod.append(temp_y)
            elif co.type != '0':
                raise RuntimeError('Unexpected cone type (' + str(co.type) + ').')
            start_row = stop_row
        y_mod = np.hstack(y_mod)
        y_mod = Expression(y_mod)
        # Now we can pretend all nonzero cones are self-dual.
        A_vals, A_rows, A_cols = [], [], []
        cur_K = [Cone(co.type, co.len) for co in self.K if co.type != '0']
        cur_K_size = sum([co.len for co in cur_K])
        if cur_K_size > 0:
            b = np.zeros(shape=(cur_K_size,))
            for i, se in enumerate(y_mod):
                if len(se.atoms_to_coeffs) == 0:
                    b[i] = se.offset
                    A_rows.append(i)
                    A_cols.append(int(ScalarVariable.curr_variable_count()) - 1)
                    A_vals.append(0)  # make sure scipy infers correct dimensions later on.
                else:
                    b[i] = se.offset
                    A_rows += [i] * len(se.atoms_to_coeffs)
                    col_idx_to_coeff = [(a.id, c) for a, c in se.atoms_to_coeffs.items()]
                    A_cols += [atom_id for (atom_id, _) in col_idx_to_coeff]
                    A_vals += [c for (_, c) in col_idx_to_coeff]
            return [(A_vals, np.array(A_rows), A_cols, b, cur_K)]
        else:
            return [([], np.zeros(shape=(0,), dtype=int), [], np.zeros(shape=(0,)), [])]

    @staticmethod
    def project(item, K):
        from sageopt.coniclifts import MIN as CL_MIN
        item = Expression(item).ravel()
        x = Variable(shape=(item.size,))
        t = Variable(shape=(1,))
        cons = [vector2norm(item - x) <= t, DualProductCone(x, K)]
        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)
        return prob.value

    def __contains__(self, item, tol=1e-7):
        residual = DualProductCone.project(item, self.K)
        return residual < tol
