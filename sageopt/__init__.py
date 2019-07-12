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
from sageopt import coniclifts
from sageopt import relaxations
from sageopt import symbolic

from sageopt.relaxations import local_refine, local_refine_polys_from_sigs
from sageopt.relaxations import conditional_sage_data
from sageopt.relaxations import sage_feasibility, sage_multiplier_search

from sageopt.symbolic.signomials import Signomial, standard_sig_monomials
from sageopt.relaxations import sig_primal, sig_dual
from sageopt.relaxations import sig_constrained_primal, sig_constrained_dual
from sageopt.relaxations import sig_solrec

from sageopt.symbolic.polynomials import Polynomial, standard_poly_monomials
from sageopt.relaxations import poly_primal, poly_dual
from sageopt.relaxations import poly_constrained_primal, poly_constrained_dual
from sageopt.relaxations import poly_solrec

__version__ = '0.3.2'
