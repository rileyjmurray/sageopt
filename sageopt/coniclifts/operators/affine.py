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
from sageopt.coniclifts.base import Expression


def cast_res(res):
    if isinstance(res, np.ndarray) and res.dtype == object:
        return Expression(res)  # includes Variables
    else:
        return res  # should only be Expressions, ScalarExpressions, and numeric ndarrays.

# Basic mathematical functions
#   https://docs.scipy.org/doc/numpy/reference/routines.math.html


# noinspection PyShadowingBuiltins,PyProtectedMember
def sum(a, axis=None, keepdims=np._NoValue, initial=np._NoValue):
    res = np.sum(a, axis=axis, keepdims=keepdims, initial=initial)
    return cast_res(res)

# Linear Algebra


def dot(a, b):
    res = np.dot(a, b)
    return cast_res(res)


def multi_dot(arrays):
    res = np.linalg.multi_dot(arrays)
    return cast_res(res)


def inner(a, b):
    res = np.inner(a, b)
    return cast_res(res)


def outer(a, b):
    res = np.outer(a, b)
    return cast_res(res)


def tensordot(a, b, axes=2):
    """
    The default value ``axes=2`` is used for consistency with numpy.
    Usage in coniclifts is mostly with ``axes=1``.
    """
    res = np.tensordot(a, b, axes)
    return cast_res(res)


def kron(a, b):
    res = np.kron(a, b)
    return cast_res(res)


def trace(a, offset=0, axis1=0, axis2=1):
    res = np.trace(a, offset, axis1, axis2)
    return cast_res(res)


# Joining, splitting, and tiling arrays.


def block(arrays):
    res = np.block(arrays)
    return cast_res(res)


def concatenate(tuple_of_arrays, axis=0):
    res = np.concatenate(tuple_of_arrays, axis)
    return cast_res(res)


def stack(arrays, axis=0):
    res = np.stack(arrays, axis)
    return cast_res(res)


def column_stack(tup):
    res = np.column_stack(tup)
    return cast_res(res)


def hstack(tup):
    res = np.hstack(tup)
    return cast_res(res)


def vstack(tup):
    res = np.vstack(tup)
    return cast_res(res)


def dstack(tup):
    res = np.dstack(tup)
    return cast_res(res)


def split(array, indices_or_sections, axis=0):
    return [cast_res(v) for v in np.split(array, indices_or_sections, axis)]


def hsplit(array, indices_or_sections):
    return [cast_res(v) for v in np.hsplit(array, indices_or_sections)]


def vsplit(array, indices_or_sections):
    return [cast_res(v) for v in np.vsplit(array, indices_or_sections)]


def dsplit(array, indices_or_sections):
    return [cast_res(v) for v in np.dsplit(array, indices_or_sections)]


def array_split(array, indices_or_sections, axis=0):
    return [cast_res(v) for v in np.array_split(array, indices_or_sections, axis)]


def tile(a, reps):
    res = np.tile(a, reps)
    return cast_res(res)


def repeat(a, repeats, axis=None):
    res = np.repeat(a, repeats, axis)
    return cast_res(res)


# array creation (from data)


def diag(v, k=0):
    res = np.diag(v, k)
    return cast_res(res)


def diagflat(v, k=0):
    res = np.diagflat(v, k)
    return cast_res(res)


def tril(m, k=0):
    res = np.tril(m, k)
    return cast_res(res)


def triu(m, k=0):
    res = np.triu(m, k)
    return cast_res(res)
