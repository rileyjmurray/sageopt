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
from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.constraints.set_membership.product_cone import PrimalProductCone, DualProductCone
from sageopt.coniclifts.base import Variable, Expression
from sageopt.coniclifts.cones import Cone
from sageopt.coniclifts.operators import affine as aff
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.operators.precompiled.relent import sum_relent, elementwise_relent
from sageopt.coniclifts.operators.precompiled import affine as comp_aff
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.standards.constants import minimize as CL_MIN, solved as CL_SOLVED
import numpy as np
import warnings
from scipy.sparse import issparse
import scipy.special as special_functions

_ALLOWED_CONES_ = {'+', 'S', 'e', '0'}

_AGGRESSIVE_REDUCTION_ = True

_ELIMINATE_TRIVIAL_AGE_CONES_ = True

_REDUCTION_SOLVER_ = 'ECOS'


def check_cones(K):
    if any([co.type not in _ALLOWED_CONES_ for co in K]):
        raise NotImplementedError()
    pass

