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


def patch_scipy_array_priority():
    """Monkey-patch scipy.sparse to make it respect __array_priority__.

    This works around outstanding issues that remain after
    https://github.com/scipy/scipy/issues/4819 was resolved, and is adapted from
    https://github.com/scipy/scipy/issues/4819#issuecomment-920722279
    to the latest version of scipy.
    """
    import scipy.sparse

    def teach_array_priority(operator):
        def respect_array_priority(self, other):
            if hasattr(other, "__array_priority__") \
            and self.__array_priority__ < other.__array_priority__:
                return NotImplemented
            else:
                return operator(self, other)

        return respect_array_priority

    def get_sparse_matrix_types():
        base_types = [scipy.sparse.spmatrix]
        try:
            base_types.append(scipy.sparse._base._spbase)
        except:  # noqa: E722
            pass
        base_types = tuple(base_types)
        concrete_types = []
        for val in scipy.sparse.__dict__.values():
            if not isinstance(val, type):
                continue
            if issubclass(val, base_types) and val not in base_types:
                concrete_types.append(val)
        
        concrete_types = tuple(concrete_types)
        return concrete_types

    concrete_types = get_sparse_matrix_types()
    BINARY_OPERATIONS = (
        "__add__", "__div__", "__eq__", "__ge__", "__gt__", "__le__",
        "__lt__", "__matmul__", "__mul__", "__ne__", "__pow__", "__sub__",
        "__truediv__",
    )

    for matrix_type in concrete_types:
        for operator_name in BINARY_OPERATIONS:
            if hasattr(matrix_type, operator_name):
                operator = getattr(matrix_type, operator_name)
                wrapped_operator = teach_array_priority(operator)
                setattr(matrix_type, operator_name, wrapped_operator)

patch_scipy_array_priority()
