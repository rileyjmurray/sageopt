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

class Constraint(object):

    def is_affine(self):
        raise NotImplementedError()

    def is_elementwise(self):
        raise NotImplementedError()

    def is_setmem(self):
        raise NotImplementedError()

    def variables(self):
        raise NotImplementedError()

    def conic_form(self):
        """
        :return: A_vals, A_rows, A_cols, b, K, sep_K
            A_vals - list (of floats)
            A_rows - numpy 1darray (of integers)
            A_cols - list (of integers)
            b - numpy 1darray (of floats)
            K - list (of coniclifts Cone objects)
            sep_K - list (of appropriately annotated coniclifts Cone objects)
        """
        raise NotImplementedError()

    def violation(self):
        raise NotImplementedError()
