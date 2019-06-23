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
from itertools import product
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

__REAL_TYPES__ = (int, float, np.int_, np.float_, np.float128)


def zero_func():
    return 0


def array_index_iterator(shape):
    return product(*[range(d) for d in shape])


def find_zero_cols_outside_range(A, keepers):
    # A is a sp matrix.
    # We want to identify indices of columns of A
    # that are all zeros, and *not* in the list called "keepers".
    bool_drop_col = np.ones((A.shape[1],), dtype=bool)
    bool_drop_col[np.hstack(keepers)] = False
    _, nonzero_col_idxs = A.nonzero()
    nonzero_col_idxs = np.unique(nonzero_col_idxs)
    bool_drop_col[nonzero_col_idxs] = False
    return np.where(bool_drop_col)[0]


def remove_select_cols_from_matrix(A, select_zero_cols):
    # A is a csc sp matrix.
    # We want to modify A in such a way that certain
    # columns of all zeros (as found in select_zero_cols)
    # never existed.
    #
    # We also want to return a vector "mapping" so that if "idxs"
    # is a numpy array of integers describing columns that
    # were NOT dropped, then "mapping[idxs]" gives the positions
    # of those column indices after being appropriately shifted
    # leftward. For performance reasons "mapping" will be a vector
    # of length A.shape[1] (for the original matrix A),
    # even though this will result in "mapping" having some meaningless
    # entries.
    zcs = set(select_zero_cols.tolist())
    mapping = np.arange(A.shape[1])
    for i in range(A.shape[1]):
        if i in zcs:
            continue
        else:
            mapping[i] -= np.count_nonzero(select_zero_cols < i)
    cols_to_keep = np.ones((A.shape[1] + 1,), dtype=bool)
    cols_to_keep[select_zero_cols] = False
    A.indptr = A.indptr[cols_to_keep]
    A._shape = (A.shape[0], A.shape[1] - len(select_zero_cols))
    return A, mapping


def sparse_matrix_data_to_csc(data_tuples, num_cols=None):
    """
    :param data_tuples: a list of quadruplets, each of which contains the data necessary to
    construct a scipy sp matrix.
    :param num_cols: the number of columns to require of the returned matrix (this parameter
    is important if some trailing columns of the matrix are zero, but we want them nevertheless).

    :return: a CSC matrix that is equivalently formed by concatenating the various CSC matrices
    specified by the quadruplets in "data_tuples".
    """
    # d in data_tuples is length 4, and has the following format:
    #   d[0] = A_vals, a list
    #   d[1] = A_rows, a 1d numpy array
    #   d[2] = A_cols, a list
    #   d[3] = the number of rows of this matrix block
    A_cols = []
    A_vals = []
    row_index_offset = 0
    for A_v, A_r, A_c, num_rows in data_tuples:
        A_r[:] += row_index_offset
        A_cols += A_c
        A_vals += A_v
        row_index_offset += num_rows
    A_rows = np.hstack([d[1] for d in data_tuples]).astype(int)
    num_rows = np.max(A_rows) + 1
    if num_cols is None:
        num_cols = np.max(A_cols) + 1
    if all(v == 0 for v in A_vals):
        A = sp.csc_matrix((int(num_rows), int(num_cols)))
    else:
        A = sp.csc_matrix((A_vals, (A_rows, A_cols)),
                          shape=(int(num_rows), int(num_cols)), dtype=float)
    return A


def parse_cones(K):
    """
    :param K: a list of Cones

    :return: a map from cone type to indices for (A,b) in the conic system
    {x : A @ x + b \in K}, and from cone type to a 1darray of cone lengths.
    """
    m = sum(co.len for co in K)
    type_selectors = defaultdict(lambda: (lambda: np.zeros(shape=(m,), dtype=bool))())
    type_to_cone_start_stops = defaultdict(lambda: list())
    running_idx = 0
    for i, co in enumerate(K):
        type_selectors[co.type][running_idx:(running_idx+co.len)] = True
        type_to_cone_start_stops[co.type] += [(running_idx, running_idx + co.len)]
        running_idx += co.len
    return type_selectors, type_to_cone_start_stops


def sparse_matrix_is_diag(A):
    nonzero_row_idxs, nonzero_col_idxs = A.nonzero()
    nonzero_col_idxs = np.unique(nonzero_col_idxs)
    nonzero_row_idxs = np.unique(nonzero_row_idxs)
    return A.nnz == A.shape[0] and len(nonzero_col_idxs) == len(nonzero_row_idxs)
