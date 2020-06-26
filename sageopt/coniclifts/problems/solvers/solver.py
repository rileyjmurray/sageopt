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


class Solver(object):
    """
    This is currently only an interface, and contains no executable code.
    """

    @staticmethod
    def apply(c, A, b, K, params):
        raise NotImplementedError()

    @staticmethod
    def solve_via_data(data, params):
        raise NotImplementedError()

    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        raise NotImplementedError()

    @staticmethod
    def is_installed():
        raise NotImplementedError()
