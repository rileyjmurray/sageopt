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
import copy
import numpy as np
import scipy.sparse as sp
from sageopt.coniclifts.cones import Cone


def separate_cone_constraints(A, b, K, dont_sep=None):
    """
    Given an affine representation {x : A @ x + b \in K}, construct a direct
    representation {x : \exists y with  G @ [x,y] + h \in K1, y \in K2},
    where ``K1`` consists exclusively of cones types in ``dont_sep``.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        Has ``m`` rows and ``n`` columns.
    b : ndarray
        Of size ``m``.
    K : List[coniclifts.cones.Cone]
        Satisfies ``sum([co.len for co in K]) == m``.
    dont_sep : set of coniclifts.cones.Cone
        Cones that may be retained in the "affine part" of the feasible set's
        direct representation.

    Returns
    -------

    Examples
    --------

    EX1: Suppose the conic system (A,b,K) can be expressed as {x : B @ x == d, G @ x + h >=0, x >=0 }, where G
    is a non-square matrix. Then by calling this function with dont_sep=None, we obtain the conic system
       { [x,y] : A1 @ [x, y] == b0 } \cap {[x, y] : x >= 0, y >= 0}
    where A1 = [[B, 0], [G, -I]] and b0 = [d, -h]. This conic system is equivalent to (A,b,K) once we
    project onto the x coordinates.

    This case is interesting because it shows that a cone constraint "x >= 0" can be separated from
    from the matrix system, even though "x" appears in th inequality constraint "G @ x + d >= 0".


    EX2: Suppose the conic system (A,b,K) can be expressed as {x : B @ x == d, L <= x <= U }.
    Then by calling this function with dont_sep=None, we obtain the conic system
    { [x,y,z] : A1 @ [x,y,z] == b0 } \cap {[x,y,z] : y >= 0, z >= 0 }
    where A1 = [[B, 0, 0], [I, -I, 0], [-I, 0, -I]] and b0 = [b, L, -U]. This conic system is equivalent
    to (A,b,K) once we project out the [y,z] components.

    NOTE: here we ended up introducing slack variables for both inequalities in the variable "x", rather
    than writing {[x,w] : B @ x == b + B @ L, U - L - x - w == 0} \cap {[x,w] : x >= 0, w >= 0 }. This
    Second option is still a valid formulation, but it is not one that this function would generate.


    EX3: Suppose the conic system (A,b,K) can be expressed as  { x : B @ x == d, G @ x <=_{K_prime} h },
    where every entry of G is nonzero, A is not square, and K_prime does not contain the zero cone.
    Then dont_sep={'0'} yields
    { [x, y] : A1 @ [x, y] == b0 } \cap { [x, y] : y \in K_{prime} }
    for A1 = [[B, 0], [-G, -I]] and b0 = [d, -h].

    Notes
    -----
    dont_sep=None is equivalent to dont_sep={'0'}.

    Common use-case: Passing dont_sep={'0','+'}. This ensures that any constraints on nonlinear cones
            (i.e. exp, psd, soc, etc...) are stated simply as "x[indices_j] \in K_j", where
            "indices_{j1}" and "indices_{j2}" are disjoint whenever j1 \neq j2. This is the main
            modeling requirement for MOSEK's standard form.

    Alternative use-case: Passing dont_sep={'+','0','e','S'} for a conic system involving semidefinite
            constraints. This will force all of the "SDP-ness" of the conic system into disjoint slack variables,
            while aspects of the conic system regarding other cones (e.g. second order cones, exponential cones,
            nonnegative orthants, etc...) are left alone.

    """
    K = copy.copy(K)
    if dont_sep is None:
        dont_sep = set('0')
    allowed = {'0'}.union(dont_sep)  # the zero cone is never replaced by slack variables
    running_row_idx = 0
    running_new_var_idx = 0
    slacks_K = []
    aug_data = [[], [], []]  # the constituent lists will store values, rows, and columns
    for i in range(len(K)):
        co_type, co_len = K[i].type, K[i].len
        if co_type not in allowed:
            new_col_idxs = np.arange(running_new_var_idx, running_new_var_idx + co_len)
            running_new_var_idx += co_len
            aug_data[2].append(new_col_idxs)
            new_slack_co = Cone(co_type, co_len, {'col mapping': A.shape[1] + new_col_idxs})
            slacks_K.append(new_slack_co)
            aug_data[1].append(np.arange(running_row_idx, running_row_idx + co_len))
            K[i] = Cone('0', co_len)
        running_row_idx += co_len
    if running_new_var_idx > 0:
        aug_data[2] = np.hstack(aug_data[2])
        aug_data[1] = np.hstack(aug_data[1])
        aug_data[0] = -1 * np.ones(len(aug_data[1]))
        augmenting_matrix = sp.csc_matrix((aug_data[0], (aug_data[1], aug_data[2])),
                                          shape=(A.shape[0], running_new_var_idx))
        A = sp.hstack([A, augmenting_matrix], format='csc')
    return A, b, K, slacks_K


def dualize_problem(c, A, b, Kp):
    """
    min{ c @ x : A @ x + b in K} == max{ -b @ y : c = A.T @ y, y in K^\dagger }

    Parameters
    ----------
    c : ndarray with ndim == 1
    A : csc_matrix
    b : ndarray with ndim == 1
    Kp : list of Cone

    Returns
    -------
    f : ndarray with ndim == 1
    G : csc_matrix
    h : ndarray with ndim == 1
    Kd : list of Cone

    Notes
    -----
    Temporary implementation. Might end up needing to transform A, so that the
    dual problem can be stated exclusively with primal cones.
    """
    Kd = []
    for Ki in Kp:
        if Ki.type == 'e':
            Kd.append(Cone('de', 3))  # dual exponential cone
        elif Ki.type == '0':
            Kd.append(Cone('fr', Ki.len))  # free cone
        else:
            Kd.append(Ki)  # remaining cones are self-dual
    f = -b
    G = A.T
    h = c
    return f, G, h, Kd
