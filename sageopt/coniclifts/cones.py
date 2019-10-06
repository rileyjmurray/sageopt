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


class Cone(object):

    def __init__(self, cone_type, length, annotations=None):
        self.type = cone_type
        self.len = length
        if annotations is None:
            annotations = dict()
        self.annotations = annotations

    def __eq__(self, other):
        if isinstance(other, Cone):
            return self.type == other.type and self.len == other.len and self.annotations == other.annotations
        else:  # pragma: no cover
            return False
