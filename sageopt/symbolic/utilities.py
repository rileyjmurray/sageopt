"""
   Copyright 2020 Riley John Murray

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
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sageopt import coniclifts as cl
from sageopt.coniclifts.base import __REAL_TYPES__


class Labeler(object):

    def __init__(self, ct):
        self.up_next = ct

    def next_label(self):
        c = self.up_next
        self.up_next += 1
        return c


def align_basis_matrices(mats):
    mat0 = mats[0]
    aligned_rows = [ri for ri in mat0]  # initial value
    lifting_locs = [[idx for idx in range(mat0.shape[0])]]
    d0 = {tuple(ri): i for (i, ri) in enumerate(mat0)}
    labeler = Labeler(mat0.shape[0])
    acd = defaultdict(labeler.next_label, d0)
    for mat in mats[1:]:
        curr_coeff_locs = []
        for ri in mat:
            up_next = labeler.up_next
            idx = acd[tuple(ri)]
            curr_coeff_locs.append(idx)
            if idx == up_next:
                aligned_rows.append(ri)
        lifting_locs.append(curr_coeff_locs)
    for i,r in enumerate(aligned_rows):
        assert isinstance(r, np.ndarray), f'row {i} is a {type(r)}, but we require a numpy array.'
        assert r.dtype  in __REAL_TYPES__, f'row {i}\'s contents are not recognized as real numbers.'
    aligned_mat = np.vstack(aligned_rows)
    return aligned_mat, lifting_locs


def lift_basis_coeffs(coeff_vecs, lifting_locs, lift_dim):
    lifted_vecs = []
    for i, crs in enumerate(lifting_locs):
        c = coeff_vecs[i]
        if isinstance(c, cl.Expression):
            lifted_c = cl.Expression(np.zeros(lift_dim))
            lifted_c[crs] = c
        else:
            curr_dim = len(crs)
            P = sp.csr_matrix((np.ones(curr_dim), (crs, np.arange(curr_dim))),
                              shape=(lift_dim, curr_dim))
            lifted_c = P @ c
        lifted_vecs.append(lifted_c)
    return lifted_vecs


def consolidate_basis_funcs(alpha, c):
    alpha_reduced, inv, counts = np.unique(alpha, axis=0, return_inverse=True, return_counts=True)
    if np.all(counts == 1):
        # then all exponent vectors are unique
        return alpha, c
    m_reduced = alpha_reduced.shape[0]
    reducer_cols = []
    for i in range(m_reduced):
        # this way of constructing "reducer_cols" is inefficient
        idxs = np.nonzero(inv == i)[0]
        reducer_cols.append(idxs)
    if isinstance(c, cl.Expression):
        c_reduced = cl.Expression([sum(c[rc]) for rc in reducer_cols])
        # ^ should be much faster than the sparse-matrix multiply, used below.
    else:
        reducer_cols = np.hstack(reducer_cols)
        reducer_rows = np.repeat(np.arange(m_reduced), counts)
        reducer_coeffs = np.ones(reducer_rows.size)
        R = sp.csr_matrix((reducer_coeffs, (reducer_rows, reducer_cols)))
        c_reduced = R @ c
    return alpha_reduced, c_reduced


def find_zero_entries(c):
    to_drop = []
    if isinstance(c, cl.Expression):
        for i, ci in enumerate(c):
            if ci.is_constant() and ci.value == 0:
                to_drop.append(i)
    elif isinstance(c, np.ndarray) and c.dtype in __REAL_TYPES__:
        to_drop = list(np.nonzero(c == 0)[0])
    else:
        raise ValueError(f'Argument {c} must be a coniclifts Expression or a numpy array with real entries.')
    return to_drop
