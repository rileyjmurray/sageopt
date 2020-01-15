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

#TODO: create a PR for SciPy, where scipy.sparse.base.spmatrix.__matmul__
# and scipy.sparse.base.spmatrix.__rmatmul__ check __array_priority__ before
# proceding with their own implementations. Until then, we need to override
# their implementations with this helper file.

# REMARK: this file might not play well with CVXPY (which has a similar file).


from scipy.sparse.base import spmatrix
from sageopt.coniclifts import Expression as cl_exp

EXPR_CLASSES = [cl_exp]
try:
    from cvxpy.expressions.expression import Expression as cp_exp
    EXPR_CLASSES.append(cp_exp)
except ImportError:
    pass
EXPR_CLASSES = tuple(EXPR_CLASSES)

BIN_OPS = ["__div__", "__mul__", "__add__", "__sub__",
           "__le__", "__eq__", "__lt__", "__gt__", "__matmul__", "__rmatmul__"]


def wrap_bin_op(method):
    """Factory for wrapping binary operators.
    """
    def new_method(self, other):
        if isinstance(other, EXPR_CLASSES):
            return NotImplemented
        else:
            return method(self, other)
    return new_method


for method_name in BIN_OPS:
    method = getattr(spmatrix, method_name)
    new_method = wrap_bin_op(method)
    setattr(spmatrix, method_name, new_method)
