import numpy as np
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.base import Expression, ScalarVariable, Variable
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.problems.problem import Problem


class PowCone(SetMembership):
    """
    A power cone constraint is defined with an n-dimensional vector
    :math:'\alpha' which is elementwise positive and
    sums to 1. The cone can be summarized by writing that
    :math:'(w, z) \\in C_{\\mathrm{power}}' if:

    .. math::
        \\prod_{i=0}^n w_i^\\alpha_i > \\| z \\|

    where :math:'w' is an n dimensional vector and :math:'z' is a scalar.

    Parameters
    ----------

    w : Expression

        An (n+1)-dimensional vector subject to this Power Cone constraints.
        One element of this vector is z which is
        determined by the elements of

    lamb: ndarray
        An (n+1)-dimensional array with 1 negative entry and n positive entries
        whose sum is 0. The n positive entries determine the elements of
        :math:'\\alpha' while the negative entry of lamb determines the index of
        z within the w parameter.


    Attributes
    ----------
    id: int
        The id of the Power Cone constraint object

    w : Expression
        The inputted (n+1)-dimensional vector inputted
        into the Power Cone constructor

    lamb : ndarray
        The inputted (n+1)-dimensional array inputted into the
        Power Cone constructor that sums to 0

    w_low : Expression
        An n-dimensional expression used to describe w in the Power Cone

    z_low : Expression
        A scalar expression used to describe z in the Power Cone

    alpha : ndarray
        The n-dimensional vector :math:'\\alpha' that sums to 1
        that defines the power Cone constraint

    Notes
    -----

    The constructor can raise a RuntimeError if the constraint is deemed infeasible.
    """
    _CONSTRAINT_ID_ = 0

    def __init__(self, w, lamb):

        self.id = PowCone._CONSTRAINT_ID_
        PowCone._CONSTRAINT_ID_ += 1

        # Finds indices of all positive elements of lambs (alpha)
        pos_idxs = lamb > 0

        # Error if lamb and w are not same size
        if not w.size == lamb.size:
            msg = 'Incompatible dimensions for w (%s) and alpha (%s)' % (w.size, lamb.size)
            raise ValueError(msg)

        # Error if alpha has no negative number
        if np.all(pos_idxs):
            msg = 'No negative number in inputted lamb array'
            raise ValueError(msg)

        neg_idxs = lamb < 0
        if np.sum(lamb) != 0:
            msg = 'lamb does not have sum of 0'
            raise ValueError(msg)

        # Get positive values and normalize
        alpha_low = lamb[pos_idxs]/np.abs(lamb[neg_idxs])

        # Negative number corresponds to z in power cone and rest of numpy array is w
        w_low = w[pos_idxs]
        z_low = w[neg_idxs]

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

    def variables(self):
        """
        Return a list of all Variable objects appearing in this Power Cone constraint object

        Returns
        -------
        - A list of all the variables involved in the Power Cone constraint of this object
        """
        var_ids = set()
        var_list = []

        # Grab non duplicate elements from w
        for se in self.w_low.flat():
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

    def conic_form(self):
        """
        Returns a sparse matrix representation of the Power cone object. It represents of the object as
        :math:'Ax+b \\in K' where K is a cone object

        Returns
        -------
        A_vals - list (of floats)
        A_rows - numpy 1darray (of integers)
        A_cols - list (of integers)
        b - numpy 1darray (of floats)
        K - list (of coniclifts Cone objects)
        """
        A_rows, A_cols, A_vals = [], [], []
        K = [Cone('pow', self.w.size, annotations={'weights': self.alpha.tolist()})]
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
        # Loop through w and add every variable
        for se in self.z_low.flat:
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

        return [(A_vals, np.array(A_rows), A_cols, b, K)]

    @staticmethod
    def project(item, alpha):
        """
        Calculates the shortest distance (the projection) of a vector to a cone parametrized by :math:'\\alpha'
        Parameters
        ----------
        item - the point we are projecting
        alpha - the :math:'\\alpha' parameter for the Cone that we are projecting to

        Returns
        -------
        The distance of the projection to the Cone
        """
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

    def violation(self, rough=False):
        """
        Returns the violation of stored w and z to being in the Power cone parametrized by alpha. If rough = True, then
        measures the violation based off the maximum violation of the condition of the Power Cone. If not, it is based
        off the projection distance between the point and Cone.

        Parameters
        ----------
        rough - whether to use a rough approximation of the violation

        Returns
        -------
        The value of the violation of this object's w and z
        """
        if rough:
            return np.max([np.abs(self.z_low.value) - np.prod(np.power(self.w_low.value, self.alpha)), 0])
        else:
            dist = PowCone.project(self.w, self.lamb)
            return dist
