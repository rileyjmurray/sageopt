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
from sageopt.coniclifts.base import Expression, Variable
from sageopt.coniclifts.cones import Cone


def sum_relent(x, y, z, aux_vars):
    # represent "sum{x[i] * ln( x[i] / y[i] )} + z <= 0" in conic form.
    # return the Variable object created for all epigraphs needed in
    # this process, as well as A_data, b, and K.
    x, y = _align_args(x, y)
    if not isinstance(z, Expression):
        z = Expression(z)
    if z.size != 1 or not z.is_affine():
        raise RuntimeError('Illegal argument to sum_relent.')
    else:
        z = z.item()  # gets the single element
    num_rows = 1 + 3 * x.size
    b = np.zeros(num_rows,)
    K = [Cone('+', 1)] + [Cone('e', 3) for _ in range(x.size)]
    A_rows, A_cols, A_vals = [], [], []
    # populate the first row
    z_id2co = [(a.id, co) for a, co in z.atoms_to_coeffs.items()]
    A_cols += [aid for aid, _ in z_id2co]
    A_vals += [-co for _, co in z_id2co]
    aux_var_ids = aux_vars.scalar_variable_ids
    A_cols += aux_var_ids
    A_vals += [-1] * len(aux_var_ids)
    A_rows += [0] * len(A_vals)
    b[0] = -z.offset
    # populate the epigraph terms
    curr_row = 1
    _fast_elemwise_data(A_rows, A_cols, A_vals, b, x, y, aux_var_ids, curr_row)
    return A_vals, np.array(A_rows), A_cols, b, K


def elementwise_relent(x, y, z):
    """
    Return variables "z" and conic constraint data for the system
        x[i] * ln( x[i] / y[i] ) <= z[i]

    A_vals - a list of floats,
    np.array(A_rows) - a numpy array of ints,
    A_cols - a list of ints,
    b - a numpy 1darray,
    K - a list of coniclifts Cone objects (of type 'e'),
    """
    x, y = _align_args(x, y)
    num_rows = 3 * x.size
    b = np.zeros(num_rows,)
    K = [Cone('e', 3) for _ in range(x.size)]
    A_rows, A_cols, A_vals = [], [], []
    curr_row = 0
    if isinstance(z, Variable):
        aux_var_ids = z.scalar_variable_ids
        _fast_elemwise_data(A_rows, A_cols, A_vals, b, x, y, aux_var_ids, curr_row)
    else:
        z = z.ravel()
        _compact_elemwise_data(A_rows, A_cols, A_vals, b, x, y, z, curr_row)
    return A_vals, np.array(A_rows), A_cols, b, K


def _fast_elemwise_data(A_rows, A_cols, A_vals, b, x, y, aux_var_ids, curr_row):
    # aux_var_ids is a list of ScalarVariable ids for epigraph terms
    for i in range(x.size):
        # first entry of exp cone
        A_rows.append(curr_row)
        A_cols.append(aux_var_ids[i])
        A_vals.append(-1)
        # third entry of exp cone
        id2co = [(a.id, co) for a, co in x[i].atoms_to_coeffs.items()]
        A_rows += [curr_row + 2] * len(id2co)
        A_cols += [aid for aid, _ in id2co]
        A_vals += [co for _, co in id2co]
        b[curr_row + 2] = x[i].offset
        # second entry of exp cone
        id2co = [(a.id, co) for a, co in y[i].atoms_to_coeffs.items()]
        A_rows += [curr_row + 1] * len(id2co)
        A_cols += [aid for aid, _ in id2co]
        A_vals += [co for _, co in id2co]
        b[curr_row + 1] = y[i].offset
        curr_row += 3


def _compact_elemwise_data(A_rows, A_cols, A_vals, b, x, y, z, curr_row):
    # z can be any affine coniclifts Expression
    for i in range(x.size):
        # first entry of exp cone
        id2co = [(a.id, co) for a, co in z[i].atoms_to_coeffs.items()]
        A_rows += [curr_row] * len(id2co)
        A_cols += [aid for aid, _ in id2co]
        A_vals += [-co for _, co in id2co]
        b[curr_row] = -z[i].offset
        # third entry of exp cone
        id2co = [(a.id, co) for a, co in x[i].atoms_to_coeffs.items()]
        A_rows += [curr_row + 2] * len(id2co)
        A_cols += [aid for aid, _ in id2co]
        A_vals += [co for _, co in id2co]
        b[curr_row + 2] = x[i].offset
        # second entry of exp cone
        id2co = [(a.id, co) for a, co in y[i].atoms_to_coeffs.items()]
        A_rows += [curr_row + 1] * len(id2co)
        A_cols += [aid for aid, _ in id2co]
        A_vals += [co for _, co in id2co]
        b[curr_row + 1] = y[i].offset
        curr_row += 3


def _align_args(x, y):
    if not isinstance(x, Expression):
        x = Expression(x)
    if not isinstance(y, Expression):
        y = Expression(y)
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise RuntimeError('Illegal arguments to sum_relent.')
    return x, y
