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
    type_selectors = defaultdict(lambda: (lambda: np.zeros(m, dtype=bool))())
    running_idx = 0
    for i, co in enumerate(K):
        type_selectors[co.type][running_idx:(running_idx+co.len)] = True
        running_idx += co.len
    return type_selectors


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
        aug_data[0] = -1 * np.ones(len(aug_data[1]))
        augmenting_matrix = sp.csc_matrix((aug_data[0], (aug_data[1], aug_data[2])),
                                          shape=(A.shape[0], running_new_var_idx))
        A = sp.hstack([A, augmenting_matrix], format='csc')
    return A, K, slacks_K


def separate_cone_constraints(A, b, K, destructive=False, dont_sep=None):
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
    A0 = A
    K0 = K
    b0 = b
    sep_K0 = []
    #   introduce slack variables (and separately record constraints) to factor remaining constraints
    #   of type "dont_sep" out of the matrix and into disjoint diagonal homogeneous cone constraints.
    A1, K1, sep_K1 = replace_nonzero_cones_with_zero_cones_and_slacks(A0, K0, destructive=True, dont_rep=dont_sep)
    #   A1 has the same number of rows of A0, but likely has more columns.
    #   K1 is of the same length of K0, but only contains the cones in "dont_sep".
    #   sep_K1 defines the constraints on any slack variables which allowed us to replace K0 by K1 as above.
    sep_K = sep_K0 + sep_K1
    return A1, b0, K1, sep_K


def separated_cones_to_matrix_cones(sep_K, num_cols, destructive=False):
    m = sum(co.len for co in sep_K)
    A_rows = np.arange(m)
    A_vals = np.ones(m)
    A_cols = np.hstack(co.annotations['col mapping'] for co in sep_K)
    if not destructive:
        # overwrite the reference to "sep_K" because we no longer need the annotations.
        sep_K = [Cone(co.type, co.len) for co in sep_K]
    else:
        for co in sep_K:
            co.annotations = dict()
    b = np.zeros(m)
    A = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=(m, num_cols), dtype=float)
    return A, b, sep_K
