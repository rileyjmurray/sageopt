import numpy as np
from scipy.sparse import coo_matrix
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.base import Variable


class ProductCone(SetMembership):

    _CONSTRAINT_ID_ = 0

    def __init__(self, A, x, b, K):
        self.id = ProductCone._CONSTRAINT_ID_
        ProductCone._CONSTRAINT_ID_ += 1
        self.A = coo_matrix(A)
        if not isinstance(x, Variable):
            raise RuntimeError('Product cones only allow coniclifts Variables as symbolic expressions.')
        self.x = x
        self.b = b
        self.K = K
        pass

    def variables(self):
        return [self.x]

    def conic_form(self):
        A_vals, A_rows, A_cols = self.A.data.tolist(), self.A.row.tolist(), self.A.col.tolist()
        if np.max(A_rows) < self.A.shape[0]-1:
            A_vals.append(0)
            A_rows.append(self.A.shape[0]-1)
            A_cols.append(0)
        sv_ids = self.x.scalar_variable_ids
        A_cols = [sv_ids[idx] for idx in A_cols]
        b = self.b
        K = self.K
        return [(A_vals, np.array(A_rows), A_cols, b, K, [])]
