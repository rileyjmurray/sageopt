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
import warnings
import numpy as np
from collections import defaultdict
from sageopt.coniclifts import utilities as util
from sageopt.coniclifts.constraints.constraint import Constraint
from sageopt.coniclifts.constraints.elementwise import ElementwiseConstraint
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.base import ScalarVariable


#
#   Core user-facing functions
#


def compile_problem(objective, constraints):
    if not objective.is_affine():
        raise NotImplementedError('The objective function must be affine.')
    # Generate a conic system that is feasible iff the constraints are feasible.
    A, b, K, variable_map, variables, svid2col = compile_constrained_system(constraints)
    # Generate the vector for the objective function.
    # c, c_offset = compile_linear_expression(objective, svid2col)
    c, c_offset = compile_objective(objective, svid2col)
    if c_offset != 0:
        warnings.warn('A constant-term ' + str(c_offset) + ' is being dropped from the objective.')
    # c = np.array(c.todense()).flatten().astype(float)
    return c, A, b, K, variable_map, variables


def compile_constrained_system(constraints):
    """
    Construct a flattened conic representation of the set of variable values which satisfy the
    constraints ("the feasible set"). Return the flattened representation of the feasible set,
    data structures for mapping the vectorized representation back to Variables, and a list of
    all Variables needed to represent the feasible set.

    The final return argument (``svid2col``) can likely be ignored by end-users.

    Parameters
    ----------
    constraints : list of coniclifts.Constraint

    Returns
    -------
    A : CSC-format sparse matrix

        The matrix appearing in the flattened representation of the feasible set.

    b : ndarray

        The offset vector appearing in the flattened representation of the feasible set.

    K : list of coniclifts Cone objects

        The cartesian product of these cones (in order) defines the convex cone appearing
        in the flattened representation of the feasible set.

    variable_map : Dict[str, ndarray]

        A map from a Variable's ``name`` field to a numpy array. If ``myvar`` is a coniclifts
        Variable appearing in the system defined by ``constraints``, then a point ``x``
        satisfying :math:`A x + b \in K` maps to a feasible value for ``myvar`` by ::

            x0 = np.hstack([x, 0])
            myvar_val = x0[variable_map[myvar.name]]

        In particular, we guarantee ``myvar.shape == variable_map[myvar.name].shape``.
        Augmenting ``x`` by zero to create ``x0`` reflects a convention that if a component of
        a Variable does not affect the constraints, that component is automatically assigned
        the value zero.

    variables : list of coniclifts.Variable

        All proper Variable objects appearing in the constraint set, including any auxiliary
        variables introduced to obtain a flattened conic system.

    svid2col : Dict[int, int]

        A map from a ScalarVariable's ``id`` to the index of the column in ``A`` where the ScalarVariable
        participates in the conic system. If the given ScalarVariable does not participate in the conic
        system, its ``id`` maps to ``-1``.

    """
    if any(not isinstance(c, Constraint) for c in constraints):
        raise RuntimeError('compile_constraints( ... ) only accepts iterables of Constraint objects.')
    # Categorize constraints (set membership vs elementwise).
    elementwise_constrs, setmem_constrs = [], []
    for c in constraints:
        if isinstance(c, SetMembership):
            setmem_constrs.append(c)
        elif isinstance(c, ElementwiseConstraint):
            elementwise_constrs.append(c)
        else:
            raise RuntimeError('Unknown argument')
    # Compile into a conic system (substituting epigraph variables as necessary)
    A, b, K, svid2col = conify_constraints(elementwise_constrs, setmem_constrs)
    check_dimensions(A, b, K)
    # Find all variables (user-defined, and auxiliary)
    variables = find_variables_from_constraints(constraints)
    var_gens = np.array([v.generation for v in variables])
    if not np.all(var_gens == var_gens[0]):
        msg1 = '\nThe model contains Variable objects of distinct "generation".\n'
        msg2 = '\nIn between constructing some of these Variables the function \n '
        msg3 = 'coniclifts.clear_variable_indices was called. Remove this function\n'
        msg4 = 'call from your program flow and try again.\n'
        raise RuntimeError(msg1 + msg2 + msg3 + msg4)
    # Construct the "variable map"
    variables.sort(key=lambda v: v.leading_scalar_variable_id())
    var_indices = []
    for v in variables:
        vids = np.array(v.scalar_variable_ids, dtype=int)
        vi = np.array([svid2col[idx] for idx in vids])
        var_indices.append(vi)
    variable_map = make_variable_map(variables, var_indices)
    return A, b, K, variable_map, variables, svid2col


#
#   Helpers (perform the heavy lifting for core user-facing functions).
#


def conify_constraints(elementwise_constrs, setmem_constrs):
    epigraph_cone_data = epigraph_substitution(elementwise_constrs)
    # Elementwise constraint expressions have now been linearized. Any
    # necessary conic constraints on epigraph variables are in "epigraph_cone_data".
    #
    # The next line simply builds data for linear [in]equality constraints
    # on the linearized versions of the elementwise constraints.
    elementwise_cone_data = [con.conic_form() for con in elementwise_constrs]
    # Elementwise constraints have now been compiled.
    #
    # Now we compile set-membership constraints. Set-membership constraints
    # have customized (possibly sophisticated) compilation functions.
    setmem_cone_data = []
    for con in setmem_constrs:
        setmem_cone_data.extend(con.conic_form())
    # All constraints have been converted to conic form.
    #
    # Now we aggregate this data to facilitate subsequent steps in compilation;
    # we organize constraints as
    #   (1) The original, given elementwise inequalities.
    #   (2) Conic constraints on any auxilliary variables introduced
    #       from epigraph transformations.
    #   (3) The conic versions of the user's "set membership" constraints.
    all_cone_data = elementwise_cone_data + epigraph_cone_data + setmem_cone_data
    matrix_data, bs, K_total = [], [], []
    for A_v, A_r, A_c, b, K in all_cone_data:
        matrix_data.append((A_v, A_r, A_c, b.size))
        bs.append(b)
        K_total.extend(K)
    A, index_map = util.sparse_matrix_data_to_csc(matrix_data)
    b = np.hstack(bs)
    return A, b, K_total, index_map


def epigraph_substitution(elementwise_constrs):
    # Do three things
    #   (1) linearize the Constraint objects by substituting epigraph variables
    #   (2) introduce conic constraints for the epigraph variables
    #   (3) return the conic constraints from (2).
    #
    # This function is not necessary for linear programs, but it shouldn't dramatically
    # slow down LP compilation time either. By calling this function even when all
    # constraints might be linear, we can skip a potentially very expensive curvature check.
    nonlin_atom_to_scalar_exprs = defaultdict(lambda: list())
    for c in elementwise_constrs:
        for se in c.expr.flat:
            for a in se.atoms_to_coeffs:
                if not isinstance(a, ScalarVariable):
                    nonlin_atom_to_scalar_exprs[a].append(se)
        c.epigraph_checked = True
    nl_cone_data = []
    for nl in nonlin_atom_to_scalar_exprs:
        x = nl.epigraph_variable
        A_vals, A_rows, A_cols, b, K = nl.epigraph_conic_form()
        for se in nonlin_atom_to_scalar_exprs[nl]:
            c = se.atoms_to_coeffs[nl]
            del se.atoms_to_coeffs[nl]
            se.atoms_to_coeffs[x] = c
        nl_cone_data.append((A_vals, A_rows, A_cols, b, K))
    return nl_cone_data


def compile_objective(objective, svid2col):
    obj_sa = objective.ravel()[0]
    offset = obj_sa.offset
    n = len([col for col in svid2col.values() if col >= 0])
    c = np.zeros(n, dtype=float)
    if len(obj_sa.atoms_to_coeffs) > 0:
        obj_data = [(svid2col[sv.id], co) for sv, co in obj_sa.atoms_to_coeffs.items()]
        obj_sv_idxs, obj_sv_coeffs = zip(*obj_data)
        if min(obj_sv_idxs) < 0:
            msg = 'The objective Expression contains a ScalarVariable which does not appear in the constraints.'
            raise ValueError(msg)
        c[np.array(obj_sv_idxs)] = obj_sv_coeffs
    return c, offset


def make_variable_map(variables, var_indices):
    """
    :param variables: a list of Variable objects with is_proper=True. These variables
    appear in some vectorized conic system represented by {x : A @ x + b \in K}.

    :param var_indices: a list of 1darrays. The i^th 1darray in this list contains
    the locations of the i^th Variable's entries with respect to the vectorized
    conic system {x : A @ x + b \in K}.

    :return: a dictionary mapping Variable names to ndarrays of indices. These ndarrays
    of indices allow the user to access a Variable's value from the vectorized conic system
    in a very convenient way.

    For example, if "my_var" is the name of a Variable with shape (10, 2, 1, 4), and "x" is
    feasible for the conic system {x : A @ x + b \in K}, then a feasible value for "my_var"
    is the 10-by-2-by-1-by-4 array given by x[variable_map['my_var']].

    NOTES:

        This function assumes that every entry of a Variable object is an unadorned ScalarVariable.
        As a result, skew-symmetric Variables or Variables with sparsity patterns are not supported
        by this very important function. Symmetric matrices are supported by this function.

        If some but not-all components of a Variable "my_var" participate in a vectorized conic
        system {x : A @ x + b \in K }, then var_indices is presumed to map these ScalarVariables
        to the number -1. The number -1 will then appear as a value in variable_map. When values
        are set for ScalarVariable objects, the system {x : A @ x + b \in K} is extended to
        { [x,0] : A @ x + b \in K}. In this way, ScalarVariables which do not participate in a
        given optimization problem are assigned the value 0.

    """
    variable_map = dict()
    for i, v in enumerate(variables):
        temp = np.zeros(v.shape)
        j = 0
        for tup in util.array_index_iterator(v.shape):
            temp[tup] = var_indices[i][j]
            j += 1
        variable_map[v.name] = np.array(temp, dtype=int)
    return variable_map


def find_variables_from_constraints(constraints):
    """
    Return a list of all "proper" Variable objects appearing in some ``c in constraints``.
    """
    variable_ids = set()
    variables = []
    for c in constraints:
        for v in c.variables():
            if v.is_proper():
                vid = id(v)
                if vid not in variable_ids:
                    variable_ids.add(vid)
                    variables.append(v)
            else:
                msg = 'Constraint \n \t' + str(c) + '\n'
                msg += 'returned an improper Variable \n \t' + str(v) + '\n'
                raise RuntimeError(msg)
    return variables


def check_dimensions(A, b, K):
    total_K_len = sum([co.len for co in K])
    if total_K_len != A.shape[0]:
        msg = 'K specifies a ' + str(total_K_len) + ' dimensional space, but A has ' + str(A.shape[0]) + ' rows.'
        raise RuntimeError(msg)
    if total_K_len != b.size:
        msg = 'K specifies a ' + str(total_K_len) + ' dimensional space, but b is of length ' + str(b.size) + '.'
        raise RuntimeError(msg)
    pass
