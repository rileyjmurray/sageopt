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

__REAL_TYPES__ = (int, float, np.int_, np.float_, np.longdouble)


def array_index_iterator(shape):
    return product(*[range(d) for d in shape])


def sparse_matrix_data_to_csc(data_tuples):
    """
    Parameters
    ----------
    data_tuples : a list of tuples
        Each tuple is of the form ``(A_vals, A_rows, A_cols, len)``, where ``A_vals`` is a
        list, ``A_rows`` is a 1d numpy array, ``A_cols`` is a list, and ``len`` is the number
        of rows in the matrix block specified by this tuple.

    Returns
    -------
    (A, index_map) - a tuple (CSC matrix, dict)
        ``A`` is the sparse matrix formed by concatenating the various CSC matrices
        specified by the quadruplets in ``data_tuples``, and dropping / reindexing rows
         per ``index_map``.
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

    unique_cols = np.sort(np.unique(A_cols))
    index_map = defaultdict(lambda: -1)
    index_map.update({c: idx for (idx, c) in enumerate(unique_cols)})
    # ^ We convert to a defaultdict with dummy value.
    #   The dummy value is used to assign values to ScalarVariable objects
    #   whose parent Variable participates in an optimization problem,
    #   even when the ScalarVariable itself does not appear in the problem.
    A_cols = np.array([index_map[ac] for ac in A_cols])
    num_rows = np.max(A_rows) + 1
    num_cols = unique_cols.size
    A = sp.csc_matrix((A_vals, (A_rows, A_cols)),
                      shape=(int(num_rows), int(num_cols)), dtype=float)
    A.eliminate_zeros()
    return A, index_map


def contiguous_selector_lengths(sel):
    """

    :param sel: a boolean array of shape (m,)
    :return:
    """
    locs = np.where(sel)[0]
    if locs.size > 0:
        auglocs = np.concatenate([locs, [-2]])
        # ^ the [-2] is just to ensure last element in "locs" is counted in the next line
        reverse_end_flags = (auglocs[1:] - auglocs[:-1]) == 1
        stop_locs = np.where(~reverse_end_flags)[0] + 1
        # ^ endpoints (exclusive) of consecutive True's in "sel"
        lengths = np.diff(np.concatenate([[0], stop_locs]))
        return lengths.tolist()
    else:
        return []


def kernel_basis(mat, tol=1e-6):
    u, s, vh = np.linalg.svd(mat)
    rank = np.count_nonzero(s > tol)
    basis = vh[rank:, :].T
    return basis
