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
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from sageopt.symbolic.signomials import Signomial
from sageopt.symbolic.polynomials import Polynomial
from sageopt.coniclifts import Expression as CL_Expr
from sageopt.coniclifts.operators import affine as aff
from sageopt.symbolic.signomials import __EXPONENT_VECTOR_DECIMAL_POINTS__


class Labeler(object):

    def __init__(self, ct):
        self.up_next = ct

    def next_label(self):
        c = self.up_next
        self.up_next += 1
        return c


def mat_sum(f1, f2):
    s = quick_sum([f1, f2])
    return s


def mat_prod(f1, f2):
    alpha1, c1 = f1.alpha, f1.c
    alpha2, c2 = f2.alpha, f2.c
    alpha1_lift = np.tile(alpha1.astype(np.float_), reps=[alpha2.shape[0], 1])
    alpha2_lift = np.repeat(alpha2.astype(np.float_), alpha1.shape[0], axis=0)
    alpha_lift = alpha1_lift + alpha2_lift
    alpha_lift = np.round(alpha_lift, decimals=__EXPONENT_VECTOR_DECIMAL_POINTS__)
    if isinstance(c1, np.ndarray) and (isinstance(c1, CL_Expr) or c1.dtype != np.dtype('O')):
        c1_lift = aff.tile(c1, reps=alpha2.shape[0])
    else:
        raise NotImplementedError()
    if isinstance(c2, np.ndarray) and (isinstance(c2, CL_Expr) or c2.dtype != np.dtype('O')):
        c2_lift = aff.repeat(c2, repeats=alpha1.shape[0])
    else:
        raise NotImplementedError()
    c_lift = c1_lift * c2_lift #TODO: update this line to be compatible with cvxpy
    #TODO: remove duplicate rows from alpha_lift, by appropriately summing the entries of c_lift.
    p = type(f1)(alpha_lift, c_lift)
    return p


def quick_sum(funcs):
    if len(funcs) == 0:
        raise ValueError()
    elif any(not isinstance(f, Signomial) for f in funcs):
        raise ValueError()
    elif len(funcs) == 1:
        return funcs[0]
    f0 = funcs[0]
    alpha = [ai for ai in f0.alpha]  # initial value
    all_crs = [[idx for idx in range(f0.m)]]
    d0 = {tuple(ai): i for (i, ai) in enumerate(f0.alpha)}
    labeler = Labeler(f0.m)
    acd = defaultdict(labeler.next_label, d0)
    for f in funcs[1:]:
        crs = []
        for ai in f.alpha:
            up_next = labeler.up_next
            idx = acd[tuple(ai)]
            crs.append(idx)
            if idx == up_next:
                alpha.append(ai)
        all_crs.append(crs)
    alpha = np.stack(alpha, axis=0)
    num_rows = alpha.shape[0]
    lifted_cs = []
    for i, crs in enumerate(all_crs):
        c = funcs[i].c
        if isinstance(c, CL_Expr):
            lifted_c = CL_Expr(np.zeros(num_rows))
            lifted_c[crs] = c
        else:
            m = len(crs)
            P = sp.csr_matrix((np.ones(m), (crs, np.arange(m))),
                              shape=(num_rows, m))
            lifted_c = P @ c
        lifted_cs.append(lifted_c)
    c = sum(lifted_cs)
    s = type(f0)(alpha, c)
    return s
