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
import warnings

"""
The purpose of this file is to change cvxpy's behavior for the multiplication operator ``*``.
CVXPY's default behavior is for ``*`` to denote matrix multplication. I need it to mean 
elementwise multiplication, for consistency with coniclifts.

The current implementation doesn't work, because it breaks the canonicalization process,
where ``*`` is used at certain times to denote matrix multiplication. For prototyping purposes,
I'm editing cvxpy files on the fly to replace these instances with ``@``. On my desktop, 
modified files are:

/home/riley/anaconda3/envs/dev36/lib/python3.6/site-packages/cvxpy/reductions/utilities.py
/home/riley/anaconda3/envs/dev36/lib/python3.6/site-packages/cvxpy/reductions/matrix_stuffing.py
/home/riley/anaconda3/envs/dev36/lib/python3.6/site-packages/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py

in cvxpy 1.1.0a2, I only needed to edit the first of those files.
"""

try:
    import cvxpy as cp
    from cvxpy.expressions.expression import Expression as cp_exp

    CVXPY_INSTALLED = True

    original_multiply = cp_exp.__mul__

    def overridden_matmul(a, b):
        if a.shape == () or b.shape == ():
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return original_multiply(a, b)

    setattr(cp_exp, '__matmul__', overridden_matmul)

    def overridden_multiply(a, b):
        return cp.multiply(a, b)

    setattr(cp_exp, '__mul__', overridden_multiply)
    msg = "\n \nSageopt has modified CVXPY's ``*`` operator. " \
          + "\nIt now applies ELEMENTWISE multiplication."

    warnings.warn(msg)

    def vstack(arg_list):
        return cp.vstack(arg_list)

except ImportError:
    CVXPY_INSTALLED = False
