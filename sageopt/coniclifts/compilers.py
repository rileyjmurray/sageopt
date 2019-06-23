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
from collections import defaultdict
from sageopt.coniclifts import utilities as util
from sageopt.coniclifts.constraints.constraint import Constraint
from sageopt.coniclifts.base import ScalarVariable, Variable

#
#   Core user-facing functions
#


def compile_problem(objective, constraints):
    if not objective.is_affine():
        raise NotImplementedError()
    # Find all user-defined Variable objects in this model.
    user_vars = find_variables_from_constraints(constraints + [objective == 0])
    # Generate a conic system that is feasible iff the constraints are feasible.
    A, b, K, sep_K, var_name_to_locs = compile_constrained_system(constraints, user_vars)
    # Find ** all ** Variable objects in the model (user-defined, or auxilliary).
    all_vars = find_variables_from_constraints(constraints + [objective == 0])
    # Generate the vector for the objective function.
    c, c_offset, _ = compile_linear_expression(objective, all_vars)
    c = np.array(c.todense()).flatten().astype(float)
    return c, A, b, K, sep_K, var_name_to_locs, all_vars


def compile_constrained_system(constraints, variables=None):
    if any(not isinstance(c, Constraint) for c in constraints):
        raise RuntimeError('compile_constraints( ... ) only accepts iterables of Constraint objects.')
    if variables is None:
        variables = find_variables_from_constraints(constraints)
    # Categorize constraints (set membership vs elementwise).
    elementwise_constrs, setmem_constrs = [], []
    for c in constraints:
        if c.is_setmem():
            setmem_constrs.append(c)
        elif c.is_elementwise():
            elementwise_constrs.append(c)
        else:
            raise RuntimeError('Unknown argument')
    A, b, K, sep_K = conify_constraints(elementwise_constrs, setmem_constrs)
    check_dimensions(A, b, K)
    variables = update_variables_from_sep_cones(variables, sep_K)
    var_gens = np.array([v.generation for v in variables])
    if not np.all(var_gens == var_gens[0]):
        msg1 = '\nThe model contains Variable objects of distinct "generation".\n'
        msg2 = '\nThis should only happen when Variable or Constraint objects are '
        msg3 = 're-used between different Problem objects.\n'
        msg4 = '\nSuch re-use of Variable or Constraint objects is not supported.\n'
        raise RuntimeError(msg1 + msg2 + msg3 + msg4)
    A, sep_K, var_name_to_locs = finalize_system(A, sep_K, variables)
    check_dimensions(A, b, K)
    return A, b, K, sep_K, var_name_to_locs


def compile_linear_expression(expr, variables):
    lin_cone_data = (expr == 0).conic_form()
    b = -lin_cone_data[3]
    matrix_data = (lin_cone_data[0], lin_cone_data[1], lin_cone_data[2], b.size)
    # The above returns data for the negated expression.
    A = -util.sparse_matrix_data_to_csc([matrix_data],
                                        num_cols=ScalarVariable.curr_variable_count())
    A, _, var_name_to_locs = finalize_system(A, [], variables)
    return A, b, var_name_to_locs


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
    matrix_data, bs, K_total, sep_K_total = [], [], [], []
    for A_v, A_r, A_c, b, K, sep_K in all_cone_data:
        matrix_data.append((A_v, A_r, A_c, b.size))
        bs.append(b)
        K_total.extend(K)
        sep_K_total.extend(sep_K)
    cur_var_count = ScalarVariable.curr_variable_count()
    A = util.sparse_matrix_data_to_csc(matrix_data, num_cols=cur_var_count)
    b = np.hstack(bs)
    return A, b, K_total, sep_K_total


def epigraph_substitution(elementwise_constrs):
    # Do three things
    #   (1) linearize the Constraint objects by introduction of epigraph variables
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
        A_vals, A_rows, A_cols, b, K, x = nl.epigraph_conic_form()
        for se in nonlin_atom_to_scalar_exprs[nl]:
            c = se.atoms_to_coeffs[nl]
            del se.atoms_to_coeffs[nl]
            se.atoms_to_coeffs[x] = c
        nl_cone_data.append((A_vals, A_rows, A_cols, b, K, []))
    return nl_cone_data


def finalize_system(A, sep_K, variables):
    # Prepare input
    if any(not isinstance(v, Variable) for v in variables):
        raise RuntimeError('Argument "variables" must be an iterable of Variable objects.')
    variables.sort(key=lambda v: v.leading_scalar_variable_id())
    # Define helper variable (for constructing the "variable map" at the end of this function).
    var_indices = []
    for v in variables:
        var_indices.append(np.array(v.scalar_variable_ids, dtype=int))
    # Find zero columns of A, which don't correspond to given ScalarVariables.
    cols_to_remove = util.find_zero_cols_outside_range(A, var_indices)
    # Remove the columns from A. Translate variable indices in var_indices and
    # cones within "sep_K" appropriately.
    if len(cols_to_remove) > 0:
        A, remaining_col_mapping = util.remove_select_cols_from_matrix(A, cols_to_remove)
        for i in range(len(var_indices)):
            var_indices[i] = remaining_col_mapping[var_indices[i]]
        for co in sep_K:
            co_cm = co.annotations['col mapping']
            co.annotations['col mapping'] = remaining_col_mapping[co_cm]
    # Make a map from Variable names to column indices where the entries of the
    # vectorized version of the variable will appear.
    variable_map = make_variable_map(variables, var_indices)
    return A, sep_K, variable_map


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
    is the 10-by-2-by-1-by-4 array given by x[user_variable_map['my_var']].

    NOTES:

        This function assumes that every entry of a Variable object is an unadorned ScalarVariable.
        As a result, skew-symmetric Variables or Variables with sparsity patterns are not supported
        by this very important function. Symmetric matrices are supported by this function.

        var_indices is constructed in finalize_system (one of the last steps of compiling a system
        Constraints into vectorized conic form).
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
    :param constraints: a list of Constraint objects.
    :return: a list of all coniclifts Variable objects appearing
    in at least one of the Constraints.
    """
    variable_ids = set()
    variables = []
    for c in constraints:
        for v in c.variables():
            if id(v) not in variable_ids:
                variable_ids.add(id(v))
                variables.append(v)
    return variables


def update_variables_from_sep_cones(variables, sep_K):
    if len(sep_K) > 0:
        variable_ids = set(id(v) for v in variables)
        updated_variables = [v for v in variables]
        for co in sep_K:
            if 'variables' not in co.annotations:
                raise RuntimeError('Improperly annotated cone.')
            co_vars = co.annotations['variables']
            for v in co_vars:
                if id(v) not in variable_ids:
                    variable_ids.add(id(v))
                    updated_variables.append(v)
        return updated_variables
    else:
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