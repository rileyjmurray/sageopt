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
from collections import defaultdict


def build_cone_type_selectors(K):
    """
    :param K: a list of Cones

    :return: a map from cone type to indices for (A,b) in the conic system
    {x : A @ x + b \in K}, and from cone type to a 1darray of cone lengths.
    """
    m = sum(co.len for co in K)
    type_selectors = defaultdict(lambda: (lambda: np.zeros(m, dtype=bool))())
    running_idx = 0
    for i, co in enumerate(K):
        type_selectors[co.type][running_idx:(running_idx+co.len)] = True
        running_idx += co.len
    return type_selectors


class Cone(object):
    """
    Cone types can be '+' for nonnegative orthant, '0' for zero-cone,
    'fr' for the free (unconstrained) cone,  'e' for exponential cone,
    'de' for dual exponential cone, 'S' for second order cone, and 'P'
    for positive semidefinite cone.
    """

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
