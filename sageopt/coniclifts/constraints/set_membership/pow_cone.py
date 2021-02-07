import numpy as np
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.base import Expression, ScalarVariable, Variable
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.problems.problem import Problem


class PowCone(SetMembership):
    _CONSTRAINT_ID_ = 0

    """
        Creates an instance of a Power Cone whose constraints are described in cone_standards.txt
        
         In this constructor, w is an (n+1)-vector and lamb is an (n+1) vector with exactly one negative 
         component.  lamb in the input corresponds to alpha in the description of a power cone.
         
         The negative number in lamb corresponds to the index of w that is z in the description of a power cone.
    """
    def __init__(self, w, lamb):
        self.id = PowCone._CONSTRAINT_ID_
        PowCone._CONSTRAINT_ID_ += 1

        # Finds indices of all positive elements of lambs (alpha)
        pos_idxs = lamb > 0

        #Error if lamb and w are not same size
        if not w.size == lamb.size:
            raise RuntimeError('Incompatible dimensions for w (' + str(w.size) + ') and alpha (' + str(lamb.size) + ')')

        # Error if alpha has no negative number
        if np.sum(pos_idxs) == np.size(lamb):
            raise ValueError('No negative number in lamb array')

        if np.sum(lamb[pos_idxs]) == lamb[lamb < 0]:
            raise ValueError('lamb does not have sum of 0')
        # Get positive values and normalize
        alpha_low = lamb[pos_idxs]/np.abs(lamb[lamb < 0])


        # Negative number corresponds to z in power cone and rest of numpy array is w
        w_low = w[pos_idxs]
        z_low = w[lamb < 0]

        if not isinstance(w_low, Expression):
            w_low = Expression(w_low)
        if not isinstance(z_low, Expression):
            z_low = Expression(z_low)

        self.w = w
        self.lamb = lamb
        self.w_low = w_low
        self.z_low = z_low
        self.alpha = alpha_low
        pass

    """
        Returns a list of all the variables asssociated with this power cone
    """
    def variables(self):

        var_ids = set()
        var_list = []

        # Grab non duplicate elements from w
        for se in self.w_low.ravel():
            for sv in se.scalar_variables():
                if id(sv.parent) not in var_ids:
                    var_ids.add(id(sv.parent))
                    var_list.append(sv.parent)

        # Iterate through z and find new variables
        for sv in self.z_low.scalar_variables():
            if id(sv.parent) not in var_ids:
                var_ids.add(id(sv.parent))
                var_list.append(sv.parent)

        return var_list

    """
        Returns the sparse form representation of the Power Cone problem (Ax = b)
    """
    def conic_form(self):
        A_rows, A_cols, A_vals = [], [], []
        K = [Cone('pow', self.w.size, annotations=self.alpha)]
        b = np.zeros(shape=(self.w.size,))

        # Loop through w and add every variable
        for i, se in enumerate(self.w_low.flat):
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

        # Make final row of matrix the information about z
        i = self.w_low.size
        if len(self.z_low.atoms_to_coeffs) == 0:
            b[i] = self.z_low.offset
            A_rows.append(i)
            A_cols.append(ScalarVariable.curr_variable_count() - 1)
            A_vals.append(0)  # make sure scipy infers correct dimensions later on.
        else:
            b[i] = self.z_low.offset
            A_rows += [i] * len(self.z_low.atoms_to_coeffs)
            col_idx_to_coeff = [(a.id, c) for a, c in self.z_low.atoms_to_coeffs.items()]
            A_cols += [atom_id for (atom_id, _) in col_idx_to_coeff]
            A_vals += [c for (_, c) in col_idx_to_coeff]

        return [(A_vals, np.array(A_rows), A_cols, b, K)]

    @staticmethod
    def project(item, alpha):
        from sageopt.coniclifts import MIN as CL_MIN

        item = Expression(item).ravel()
        w = Variable(shape=(item.size, ))
        t = Variable(shape=(1, ))

        cons = [
            vector2norm(item-w) <= t,
            PowCone(w, alpha)
        ]

        prob = Problem(CL_MIN, t, cons)
        prob.solve(verbose=False)

        return prob.value

    """
    Returns the violation of the current values of this object 
    """
    def violation(self, rough = False):
        if rough:
            return np.max([np.abs(self.z_low.value) - np.prod(np.power(self.w_low.value, self.alpha)), 0])
        else:
            dist = PowCone.project(self.w, self.lamb)
            return dist
