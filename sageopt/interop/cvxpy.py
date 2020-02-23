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
"""

try:
    import cvxpy as cp
    from cvxpy.expressions.expression import Expression as cp_exp

    CVXPY_INSTALLED = True

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
