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
import scipy.sparse as sp
from sageopt.coniclifts import utilities as util
from collections import defaultdict
from sageopt.coniclifts.cones import Cone


def build_cone_type_selectors(K):
    """
    :param K: a list of Cones

    :return: a map from cone type to indices for (A,b) in the conic system
    {x : A @ x + b \in K}, and from cone type to a 1darray of cone lengths.
    """
    m = sum(co.len for co in K)
    type_selectors = defaultdict(lambda: (lambda: np.zeros(shape=(m,), dtype=bool))())
    running_idx = 0
    for i, co in enumerate(K):
        type_selectors[co.type][running_idx:(running_idx+co.len)] = True
        running_idx += co.len
    return type_selectors


def find_cones_with_mutually_disjoint_scope(cones):
    # The runtime of this function scales quadratically in the length of the list "cones".
    # If Mosek is going to be viable for compiling cone programs of nontrivial size, then
    # this function needs to be orders of magnitude faster. In addition to changing how it's
    # written, the final function might be simple enough to be accelerated by numba or cython.
    disjoint_cone_idxs = set()
    remaining_cone_idxs = set(np.arange(len(cones)).tolist())
    for co_idx, co in enumerate(cones):
        if 'isolated' in co.annotations and co.annotations['isolated']:
            disjoint_cone_idxs.add(co_idx)
            remaining_cone_idxs.remove(co_idx)
        else:
            co_cols = co.annotations['scope']
            is_isolated = True
            for other_idx in remaining_cone_idxs:
                if other_idx == co_idx:
                    continue
                other_cols = cones[other_idx].annotations['scope']
                if len(co_cols.intersection(other_cols)) > 0:
                    is_isolated = False
                    break
            if is_isolated:
                disjoint_cone_idxs.add(co_idx)
                remaining_cone_idxs.remove(co_idx)
    disjoint_cones = []
    for i in disjoint_cone_idxs:
        disjoint_cones.append(cones[i])
    return disjoint_cones


def find_diagonal_cone_constraints(A, K):
    A = sp.csr_matrix(A)
    diag_cones = []
    for co in K:
        co_start, co_stop = co.annotations['ss']
        # Check if this cone is diagonal.
        curr_A = A[co_start:co_stop, :].tocoo()
        if 'diagonal' in co.annotations and co.annotations['diagonal']:
            co.annotations['A'] = curr_A
            diag_cones.append(co)
        elif util.sparse_matrix_is_diag(curr_A):
            co.annotations['A'] = curr_A
            diag_cones.append(co)
    return diag_cones


def prep_sep_dis_diag_cone_cons(A, b, K, dont_sep=None):
    """
    "Prepare to separate disjoint diagonal cone constraints (from the conic system (A, b, K))."

    K is a list a Cone objects.

    """
    if dont_sep is None:
        dont_sep = set()
    dont_sep = {'0'}.union(dont_sep)
    #
    # Before doing anything we annotate the cones with the information necessary
    # to place themselves relative to (A,b). Then we find the disjoint diagonal cones.
    #
    Cone.annotate_cone_positions(K)  # gives cones 'ss' and 'position' annotations
    relevant_cones = [co for co in K if co.type not in dont_sep]
    diag_cones = find_diagonal_cone_constraints(A, relevant_cones)  # gives cones an 'A' annotation
    Cone.annotate_cone_scopes(A, diag_cones)  # gives cones a 'scope' annotation.
    disjoint_diag_cones = find_cones_with_mutually_disjoint_scope(diag_cones)
    disjoint_diagonal_idxs = set([co.annotations['position'] for co in disjoint_diag_cones])
    #
    # Build the return values
    #
    K0 = [co for i, co in enumerate(K) if i not in disjoint_diagonal_idxs]
    for i in range(len(K0)):
        K0[i].annotations = dict()
    sep_K0 = []
    type_selectors = build_cone_type_selectors(K)
    scalings = np.ones(shape=(A.shape[1],))
    translates = np.zeros(shape=(A.shape[1],))
    for co in disjoint_diag_cones:
        co_start, co_stop = co.annotations['ss']
        # record normalization and translation necessary to homogenize these cones
        curr_A = co.annotations['A']  # curr_A is COO format.
        nz_vals, nz_rows, nz_cols = curr_A.data, curr_A.row, curr_A.col
        for i, col_idx in enumerate(nz_cols):
            row_idx = co_start + nz_rows[i]
            translates[col_idx] -= b[row_idx]
            scalings[col_idx] /= nz_vals[i]
        # record a map from coordinate indices in this this cone to the columns
        # (variable indices) that must belong to the cone at the coordinate index.
        # i.e., sort the entries of nz_cols according to the entries of nz_rows
        type_selectors[co.type][co_start:co_stop] = False
        col_mapping = nz_cols[np.argsort(nz_rows)]
        new_co = Cone(co.type, co.len, {'col mapping': col_mapping})
        sep_K0.append(new_co)
    # Build the final selector
    all_selectors = [sel.reshape(-1, 1) for sel in type_selectors.values()]
    all_selectors = np.hstack(all_selectors)
    row_selector = np.any(all_selectors, axis=1)
    return row_selector, K0, sep_K0, scalings, translates


def separate_disjoint_diagonal_cones(A, b, K, destructive=False, dont_sep=None):
    """
    This function's purpose ...

        coniclifts' standard form for a feasible set is {x : A @ x + b \in K}. Some solvers
        require alternative standard forms, such as {x : A @ x == b, x \in K },
        or {x : A @ x <= b, x \in K }. Mosek requires an even more peculiar form, namely
        {(x,X) : A @ x + F(X) <= b, x \in K, X >> 0 }, where F is a particular linear map.

        We can trivially convert from the coniclifts standard form to the above forms by
        introduction of slack variables. I.e. represent {x : A @ x + b \in K} as
        { x : y == A @ x + b, y \in K }. The drawback is that doing this substitution
        can substantially worsen the conditioning of the problem. This function exists
        to facilitate similar reformulation, without introducing unnecessary slack
        variables.

    :param A: a scipy csc sparse matrix (m rows, n columns)
    :param b: a numpy 1darray of length m
    :param K: a list of cones, specified by (str, int) pairs, where "str" gives the
    cones type, and "int" gives the cone's length.
    :param destructive: a boolean. Indicates whether or not inputs (A,b) are modified,
    or copied.
    :param dont_sep: a set of cone types that are not to be modified while reformulating
    the system. Common values are dont_sep={'0'} and dont_sep={'0','+'}.
    :return: See in-line comments.
    """
    if not destructive:
        A = A.copy()
        b = b.copy()
    sel, K0, sep_K0, scale, trans = prep_sep_dis_diag_cone_cons(A, b, K, dont_sep=dont_sep)
    # In order to separate disjoint diagonal cones from the system (A, b, K) as suggested by
    # the function above, we will need to apply a diagonal affine change of coordinates to
    # {x : A @ x + b \in K}. That diagonal change of coordinates maps "x" to
    # x0 == np.diag(scale) @ (x + trans).
    #
    # Since "x" and "x0" don't yet exist on the computer, we transform (A, b) to
    # (A0, b0) in such a way that the set of feasible "x" maps to {x0 : A0 @ x0 + b0 \in K}.
    b0 = b + A.dot(trans)
    A0 = A @ sp.diags(scale)
    # Now that this map has been applied, we can drop a subset of rows from (A, b)
    # --- they are now accounted for by "sep_K0". The returned value "K0" has
    # already been updated to reflect the fact that we intend to drop these rows.
    A0 = A0[sel, :]
    b0 = b0[sel]
    # Our feasible set has now been mapped to { x0 : A0 @ x0 + b0 \in K0 and x0 \in sep_K0 }.
    #
    # In order to invert this map later, we return both (A0, b0, K0, sep_K0) and the
    # transformation data given by (scale, trans).
    return A0, b0, K0, sep_K0, scale, trans


def replace_nonzero_cones_with_zero_cones_and_slacks(A, K, destructive=False, dont_rep=None):
    # It only really makes sense to call this function after any isolated diagonal cone
    # constraints have been separated / factored out of (A,K).
    if not destructive:
        A = A.copy()
        K = K.copy()
    if dont_rep is None:
        dont_rep = set()
    allowed = {'0'}.union(dont_rep)  # the zero cone is never replaced by slack variables
    running_row_idx = 0
    running_new_var_idx = 0
    slacks_K = []
    aug_data = [[], [], []]  # the constituent lists will store values, rows, and columns
    for i in range(len(K)):
        (co_type, co_len) = K[i].type, K[i].len
        if co_type not in allowed:
            new_col_idxs = np.arange(running_new_var_idx, running_new_var_idx + co_len)
            running_new_var_idx += co_len
            aug_data[2].append(new_col_idxs)
            new_slack_co = Cone(co_type, co_len, {'col mapping': A.shape[1] + new_col_idxs})
            slacks_K.append(new_slack_co)
            aug_data[1].append(np.arange(running_row_idx, running_row_idx + co_len))
            K[i] = Cone('0', co_len)
        running_row_idx += co_len
    if len(aug_data[2]) > 0:
        aug_data[2] = np.hstack(aug_data[2])
        aug_data[1] = np.hstack(aug_data[1])
        aug_data[0] = -1 * np.ones(shape=(len(aug_data[1]),))
        augmenting_matrix = sp.csc_matrix((aug_data[0], (aug_data[1], aug_data[2])),
                                          shape=(A.shape[0], running_new_var_idx))
        A = sp.hstack([A, augmenting_matrix], format='csc')
    return A, K, slacks_K


def separate_cone_constraints(A, b, K, destructive=False, dont_sep=None, avoid_slacks=False):
    """
    Replace the conic system {x : A @ x + b \in K } by {x0 : A1 @ x1 + b1 \in K}

    :param A: a scipy csc sparse matrix (m rows, n columns)
    :param b: a numpy 1darray of length m
    :param K: a list of cones, specified by (str, int) pairs, where "str" gives the
    cones type, and "int" gives the cone's length.
    :param destructive: a boolean. Indicates whether or not inputs (A,b) are modified,
    or copied.
    :param dont_sep: a set of cone types that are not to be modified while reformulating
    the system. Common values are dont_sep={'0'} and dont_sep={'0','+'}.
    :param avoid_slacks: boolean. Indicates whether or not to perform pre-processing to minimize
    the number of slack variables.

    :return: See in-line comments.

    Example: Suppose the conic system (A,b,K) can be expressed as {x : B @ x == d, G @ x + h >=0, x >=0 }, where G
             is a non-square matrix. Then by calling this function with dont_sep=None, we obtain the conic system
                { [x,y] : A1 @ [x, y] == b0 } \cap {[x, y] : x >= 0, y >= 0}
             where A1 = [[B, 0], [G, -I]] and b0 = [d, -h]. This conic system is equivalent to (A,b,K) once we
             project onto the x coordinates.

             This case is interesting because it shows that a cone constraint "x >= 0" can be separated from
             from the matrix system, even though "x" appears in th inequality constraint "G @ x + d >= 0".

    Example: Suppose the conic system (A,b,K) can be expressed as {x : B @ x == d, L <= x <= U }.
             Then by calling this function with dont_sep=None, we obtain the conic system
                { [x,y,z] : A1 @ [x,y,z] == b0 } \cap {[x,y,z] : y >= 0, z >= 0 }
             where A1 = [[B, 0, 0], [I, -I, 0], [-I, 0, -I]] and b0 = [b, L, -U]. This conic system is equivalent
             to (A,b,K) once we project out the [y,z] components.

             NOTE: here we ended up introducing slack variables for both inequalities in the variable "x", rather
             than writing {[x,w] : B @ x == b + B @ L, U - L - x - w == 0} \cap {[x,w] : x >= 0, w >= 0 }. This
             Second option is still a valid formulation, but it is not one that this function would generate.

    Example: Suppose the conic system (A,b,K) can be expressed as  { x : B @ x == d, G @ x <=_{K_prime} h },
             where every entry of G is nonzero, A is not square, and K_prime does not contain the zero cone.
             Then dont_sep={'0'} yields
                { [x, y] : A1 @ [x, y] == b0 } \cap { [x, y] : y \in K_{prime} }
             for A1 = [[B, 0], [-G, -I]] and b0 = [d, -h].

    Common use-case: Passing dont_sep={'0','+'}. This ensures that any constraints on nonlinear cones
            (i.e. exp, psd, soc, etc...) are stated simply as "x[indices_j] \in K_j", where
            "indices_{j1}" and "indices_{j2}" are disjoint whenever j1 \neq j2. This is the main
            modeling requirement for MOSEK's standard form.

    Alternative use-case: Passing dont_sep={'+','0','e','S'} for a conic system involving semidefinite
            constraints. This will force all of the "SDP-ness" of the conic system into disjoint slack variables,
            while aspects of the conic system regarding other cones (e.g. second order cones, exponential cones,
            nonnegative orthants, etc...) are left alone.

    REMARKS:
        (1) dont_sep=None is equivalent to dont_sep={'0'}.
    """
    if not destructive:
        A = A.copy()
        K = K.copy()
    if dont_sep is None:
        dont_sep = set('0')
    if avoid_slacks:
        # (1) identify "simple" nonlinear cones, and factor them out from the matrix part of the conic system.
        A0, b0, K0, sep_K0, scale, trans = separate_disjoint_diagonal_cones(A, b, K, True, dont_sep)
        #   A0 has the same number of columns as A, but likely fewer rows.
        #   K0 is no longer than K.
        #   sep_K0 contains information for block elementwise constraints on "x".
        #   scale and trans contain the data for a diagonal affine change of coordinates between our
        #   original variables {x : A @ x + b \in K} and the new variables {x0 : A0 @ x0 + b0 \in K0, x0 \in sep_K0 }.
    else:
        A0 = A
        K0 = K
        b0 = b
        sep_K0 = []
        scale = np.ones(shape=(A.shape[1],))
        trans = np.zeros(shape=(A.shape[1],))
    # (2) introduce slack variables (and separately record constraints) to factor remaining constraints
    #     of type "dont_sep" out of the matrix and into disjoint diagonal homogeneous cone constraints.
    A1, K1, sep_K1 = replace_nonzero_cones_with_zero_cones_and_slacks(A0, K0, destructive=True, dont_rep=dont_sep)
    scale = np.hstack((scale, np.ones(shape=(A1.shape[1] - len(scale)))))
    trans = np.hstack((trans, np.zeros(shape=(A1.shape[1] - len(trans)))))
    #   A1 has the same number of rows of A0, but likely has more columns.
    #   K1 is of the same length of K0, but only contains the cones in "dont_sep".
    #   sep_K1 defines the constraints on any slack variables which allowed us to replace K0 by K1 as above.
    #   The introduction of new variables required that we appropriately pad "scale" and "trans".
    sep_K = sep_K0 + sep_K1
    return A1, b0, K1, sep_K, scale, trans


def separated_cones_to_matrix_cones(sep_K, num_cols, destructive=False):
    m = sum([co.len for co in sep_K])
    A_rows = np.arange(m)
    A_vals = np.ones(shape=(m,))
    A_cols = np.hstack([co.annotations['col mapping'] for co in sep_K])
    if not destructive:
        # overwrite the reference to "sep_K" because we no longer need the annotations.
        sep_K = [Cone(co.type, co.len) for co in sep_K]
    else:
        for co in sep_K:
            co.annotations = dict()
    b = np.zeros(shape=(m,))
    A = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=(m, num_cols), dtype=float)
    return A, b, sep_K
