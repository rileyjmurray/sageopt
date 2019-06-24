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

from sageopt.symbolic.signomials import Signomial, standard_sig_monomials
from sageopt.relaxations.sage_sigs import conditional_sage_data
from sageopt.relaxations.sage_sigs import sage_primal, sage_dual, sage_feasibility, sage_multiplier_search
from sageopt.relaxations.sage_sigs import constrained_sage_primal, constrained_sage_dual
from sageopt.relaxations.sig_solution_recovery import dual_solution_recovery, local_refinement

from sageopt.symbolic.polynomials import Polynomial, standard_poly_monomials
from sageopt.relaxations.sage_polys import conditional_sage_poly_data
from sageopt.relaxations.sage_polys import sage_poly_primal, sage_poly_dual
from sageopt.relaxations.sage_polys import sage_poly_feasibility, sage_poly_multiplier_search
from sageopt.relaxations.sage_polys import constrained_sage_poly_primal, constrained_sage_poly_dual
from sageopt.relaxations.poly_solution_recovery import dual_solution_recovery as dual_poly_solution_recovery
from sageopt.relaxations.poly_solution_recovery import local_refinement as local_poly_refinement

__version__ = '0.3.0'
