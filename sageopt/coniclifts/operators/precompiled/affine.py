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
from sageopt.coniclifts.base import Variable, Expression, ScalarExpression, ScalarVariable
from sageopt.coniclifts.operators.affine import concatenate


def matvec(mat, vec):
    """
    The argument "vec" is an affine image of some coniclifts Variables "x".
    This function returns sparse data for a map "x |--> A @ x + b" which is
    equivalent to "vec |--> mat @ vec".
    Parameters
    ----------
    mat : ndarray
        Shape (m, n)
    vec : Expression
        Shape (n,)
    Returns
    -------
    A_vals : list
    A_rows : ndarray
    A_cols : list
    b : ndarray of shape (m,)
    Notes
    -----
    This function will be faster if "vec" is a coniclifts Variable object.
    """
    num_rows = mat.shape[0]
    b = np.zeros(num_rows)
    if isinstance(vec, Variable):
        vec1_ids = vec.scalar_variable_ids
    else:
        A1, x1, b1 = vec.ravel().factor()
        mat = mat @ A1
        b += mat @ b1
        vec1_ids = [xi.id for xi in x1]
    A_vals, A_rows, A_cols = _matvec_by_var_indices(mat, vec1_ids)
    return A_vals, A_rows, A_cols, b


def _matvec_by_var_indices(mat, var_ids):
    A_rows = np.tile(np.arange(mat.shape[0]), reps=mat.shape[1])
    A_cols = np.repeat(var_ids, mat.shape[0]).tolist()
    A_vals = mat.ravel(order='F').tolist()  # stack columns, then tolist
    return A_vals, A_rows, A_cols


def matvec_plus_matvec(mat1, vec1, mat2, vec2):
    # TODO: fix docstring
    """
    :param mat1: a numpy ndarray of shape (m, n1).
    :param vec1: a coniclifts Variable of shape (n1,) or (n1, 1).
    :param mat2: a numpy ndarray of shape (m, n2)
    :param vec2: a coniclifts Variable of shape (n2,) or (n2, 1).
    :return: A_vals, A_rows, A_cols so that the coniclifts Expression
        expr = mat1 @ vecvar1 + mat2 @ vecvar2 would have "expr >= 0"
        compile to A_vals, A_rows, A_cols, np.zeros((m,)), [].
    """
    mat = np.hstack([mat1, mat2])
    if isinstance(vec1, Variable) and isinstance(vec2, Variable):
        num_rows = mat.shape[0]
        b = np.zeros(num_rows)
        ids = vec1.scalar_variable_ids + vec2.scalar_variable_ids
        A_vals, A_rows, A_cols = _matvec_by_var_indices(mat, ids)
    else:
        vec = concatenate((vec1, vec2))
        A_vals, A_rows, A_cols, b = matvec(mat, vec)
    return A_vals, A_rows, A_cols, b


def matvec_minus_vec(mat, vec1, vec2):
    # TODO: fix docstring
    """
    :param mat: a numpy ndarray of shape (m, n).
    :param vec1: a coniclifts Variable of shape (n,) or (n, 1).
    :param vec2: a coniclifts Variable of shape (m,) or (m, 1).
    :return: A_vals, A_rows, A_cols so that the coniclifts Expression
        expr = mat @ vecvar1 - vecvar2 would have "expr >= 0"
        compile to A_vals, A_rows, A_cols, np.zeros((m,)), [].
    """
    num_rows = mat.shape[0]
    # get the block corresponding to the matvec
    A_vals, A_rows, A_cols, b = matvec(mat, vec1)
    # add the information for the "minus vec"
    if isinstance(vec2, Variable):
        vec2_ids = vec2.scalar_variable_ids
        A_vals += [-1.0] * num_rows
        A_rows = np.concatenate((A_rows, np.arange(num_rows)))
        A_cols += vec2_ids
    else:
        for i in range(num_rows):
            id2co = [(a.id, co) for a, co in vec2[i].atoms_to_coeffs.items()]
            A_cols += [aid for aid, _ in id2co]
            A_vals += [-co for _, co in id2co]
            A_rows += [i] * len(id2co)
        b2 = np.array([se.offset for se in vec2])
        b -= b2
    return A_vals, A_rows, A_cols, b


def matvec_plus_vec_times_scalar(mat1, vec1, vec2, scalar):
    # TODO: fix docstring
    """
    :param mat1: a numpy ndarray of shape (m, n).
    :param vec1: a coniclifts Variable of shape (n,) or (n, 1)
    :param vec2: a numpy ndarray of shape (m,) or (m, 1).
    :param scalar: a coniclifts Variable of size 1.
    :return:
    Return A_rows, A_cols, A_vals as if they were generated by a compile() call on
    con = (mat @ vecvar + vec * singlevar >= 0).
    """
    if isinstance(scalar, Expression):
        if scalar.size == 1:
            scalar = scalar.item()
        else:
            raise ValueError('Argument `scalar` must have size 1.')
    # We can now assume that "scalar" is a ScalarExpression.
    if isinstance(vec1, Variable):
        b = scalar.offset * vec2
        a2c = scalar.atoms_to_coeffs.items()
        s_covec = np.array([co for (sv, co) in a2c])
        s_indices = [sv.id for (sv, co) in a2c]
        mat2 = np.outer(vec2, s_covec)  # rank 1
        mat = np.hstack((mat1, mat2))
        indices = vec1.scalar_variable_ids + s_indices
        A_vals, A_rows, A_cols = _matvec_by_var_indices(mat, indices)
    else:
        mat = np.column_stack((mat1, vec2))
        if isinstance(scalar, ScalarExpression):
            scalar = Expression([scalar])
        expr = concatenate((vec1, scalar))
        A_vals, A_rows, A_cols, b = matvec(mat, expr)
    return A_vals, A_rows, A_cols, b


def columns_sum_leq_vec(mat, vec, mat_offsets=False):
    # This function assumes that each ScalarExpression in mat
    # consists of a single ScalarVariable with coefficient one,
    # and has no offset term.
    A_rows, A_cols, A_vals = [], [], []
    m = mat.shape[0]
    if m != vec.size:
        raise RuntimeError('Incompatible dimensions.')
    b = np.zeros(m,)
    for i in range(m):
        # update cols and data to reflect addition of elements in ith row of mat
        svs = mat[i, :].scalar_variables()
        A_cols += [sv.id for sv in svs]
        A_vals += [-1] * len(svs)
        # update cols and data to reflect addition of elements from ith element of vec
        #   ith element of vec is a ScalarExpression!
        id2co = [(a.id, co) for a, co in vec[i].atoms_to_coeffs.items()]
        A_cols += [aid for aid, _ in id2co]
        A_vals += [co for _, co in id2co]
        # update rows with appropriate number of "i"s.
        total_len = len(svs) + len(id2co)
        if total_len > 0:
            A_rows += [i] * total_len
        else:
            A_rows.append(i)
            A_vals.append(0)
            A_cols.append(ScalarVariable.curr_variable_count() - 1)
        # update b
        b[i] = vec[i].offset
        if mat_offsets:
            b[i] -= sum([matij.offset for matij in mat[i, :]])
    A_rows = np.array(A_rows)
    return A_vals, A_rows, A_cols, b
