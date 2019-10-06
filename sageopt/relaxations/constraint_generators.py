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
from itertools import combinations_with_replacement
from sageopt.symbolic.signomials import Signomial
import warnings


def up_to_q_fold_cons(cons, q):
    cons = [g for g in cons]  # shallow copy
    if len(cons) == 0 or q == 1:
        return cons
    else:
        fold_cons = set()
        for qq in range(q + 1):
            if qq == 0:
                continue
            for comb in combinations_with_replacement(cons, qq):
                temp = np.prod(comb)
                # noinspection PyUnresolvedReferences
                if np.count_nonzero(temp.c) > 1:
                    fold_cons.add(temp)
        fold_cons = list(fold_cons)
    return fold_cons


def valid_posynomial_inequalities(gs):
    conv_gs = []
    for g in gs:
        # g defines a constraint g(x) >= 0
        num_pos = np.count_nonzero(g.c > 0)
        if num_pos >= 2:
            # cannot convexify
            continue
        elif num_pos == 0 and np.count_nonzero(g.c < 0) > 0:  # pragma: no cover
            raise RuntimeError('Attempting to convexify an infeasible signomial inequality constraint.')
        else:
            pos_loc = np.where(g.c > 0)[0][0]
            inverse_term = Signomial({tuple(-g.alpha[pos_loc, :]): 1})
            conv_gs.append(g * inverse_term)
    return conv_gs


def valid_monomial_equations(eqs):
    conv_eqs = []
    for g in eqs:
        # g defines a constraint g(x) == 0.
        if np.count_nonzero(g.c) > 2:
            # cannot convexify
            continue
        pos_loc = np.where(g.c > 0)[0]
        if pos_loc.size == 1:
            pos_loc = pos_loc[0]
            inverse_term = Signomial({tuple(-g.alpha[pos_loc, :]): 1})
            conv_eqs.append(g * inverse_term)
    return conv_eqs


def valid_gp_representable_poly_inequalities(gs):
    gp_rep_polys = []
    for g in gs:
        is_even = len(g.even_locations()) == g.m
        num_pos = np.count_nonzero(g.c > 0)
        if num_pos == 1 and is_even:
            gp_rep_polys.append(g)
        elif num_pos == 0 and is_even:
            if g(np.zeros(g.n)) == 0:
                warnings.warn('The polynomial "g" is only nonnegative at the origin.')
            else:
                raise RuntimeError('Infeasible polynomial inequality.')
    return gp_rep_polys


def valid_gp_representable_poly_eqs(eqs):
    gp_rep_polys = []
    for g in eqs:
        is_even = len(g.even_locations()) == g.m
        if is_even and np.count_nonzero(g.c != 0) == 2 and np.count_nonzero(g.c > 0) == 1:
            gp_rep_polys.append(g)
    return gp_rep_polys
