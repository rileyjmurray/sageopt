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
from sageopt.symbolic.signomials import __EXPONENT_VECTOR_DECIMAL_POINTS__
from sageopt.symbolic.signomials import Signomial
from sageopt.symbolic.polynomials import Polynomial


__EXPONENT_VECTOR_TOLERANCE__ = 10**-(__EXPONENT_VECTOR_DECIMAL_POINTS__ + 1)


def relative_coeff_vector(s, reference_alpha):
    c = np.zeros(reference_alpha.shape[0])
    sc = s.c
    if hasattr(sc, 'value'):
        sc = sc.value
    common, corr = row_correspondence(s.alpha, reference_alpha)
    c[corr] = sc[common]
    return c


def row_correspondence(alpha1, alpha2):
    """
    It returns lists "common" and "alpha1_to_alpha2" such that

        alpha1[common, :] == alpha2[alpha1_to_alpha2, :].

    This is useful because it allows us to speak of the "i-th" exponent in a meaningful
    way when dealing with Signomials, without having to adopt a canonical ordering for
    exponent vectors.

    :param alpha1: numpy n-d array.
    :param alpha2: a numpy n-d array.
    :return: a list "alpha1_to_alpha2" such that alpha1 == alpha2[alpha1_to_alpha2, :].
    """
    common = []
    alpha1_to_alpha2 = []
    for i, row in enumerate(alpha1):
        # noinspection PyTypeChecker
        shifted = alpha2 - row
        locs = np.where(np.all(np.abs(shifted) < __EXPONENT_VECTOR_TOLERANCE__, axis=1))
        if len(locs[0]) > 0:
            common.append(i)
            loc = locs[0][0]
            alpha1_to_alpha2.append(loc)
    return common, alpha1_to_alpha2


def moment_reduction_array(s_h, h, L):
    """
    :param s_h: a Signomial or Polynomial (likely with coniclifts Variables as coefficients)
    :param h: a Signomial or Polynomial (with only scalars as coefficients)
    :param L: a Signomial or Polynomial (likely with coniclifts Variables as coefficients)

    Assumptions:

        The rows of lagrangian.alpha must include all rows of (s_h * h).alpha.
        s_h, h, and lagrangian are all of the same type (i.e. all Signomials, or all Polynomials).

    Primary Usage:

        A constraint "h(x) == 0" or "h(x) >= 0" appears in an optimization problem. That constraint
        has been incorporated into a Lagrangian "L", with an associated Lagrange muliplier "s_h".

        Let w(x) = s_h(x) * h(x), and let F be the nonlinear map (consisting entrywise of signomials,
        or polynomials) satisfying w(x) = s_h.c @ F(x). Such a map certainly exists, although its
        entries may include complicated functions with multiple terms (i.e. not just monomials).

        Let G be either G(x) = x^L.alpha (if L is a Polynomial) or G(x) = exp(L.alpha @ x)
        (if L is a Signomial), so that L(x) = L.c @ G(x).

        Since w.alpha is a subset of L.alpha, there exists a matrix C so that F(x) == C @ G(x),
        and in turn so that w(x) == s_h.c @ (C @ G(x)).

        This function returns that matrix C, for use in constructing dual SAGE relaxations.

        The specific usage of C is as follows. Suppose that "v" is the dual variable to the constraint
        that "lagrangian is SAGE". If s_h was a SAGE function, the dual problem will include
        a constraint of the form "C @ v is in the dual SAGE cone over exponents s_h.alpha".
        If s_h was unconstrained, then the dual problem will include a constraint "C @ v == 0".

    Secondary usage:

        We want to minimize the function s_h over some X \\subset \\R^n.
        The function "h" is fixed, and is known to be positive almost-everywhere on X.
        s_h and h are related to L by the identity L == s_h * h.

        Returns a matrix "C" that maps outer-approximations of the moment-cone for functions
        "like" L (i.e. Polynomials or Signomials over exponents L.alpha) to outer-approximations
        of the moment-cone for functions "like" s_h (Polynomials or Signomials over exponents
        s_h.alpha).
    """
    if isinstance(h, Polynomial):
        classname = Polynomial
    elif isinstance(h, Signomial):
        classname = Signomial
    else:
        raise RuntimeError('Unknown argument.')
    minimal_L = s_h * h
    for row in minimal_L.alpha_c:
        if row not in L.alpha_c:
            msg0 = 'The product s_h * h contains an exponent vector \n'
            msg1 = '\t ' + str(row) + '\n'
            msg2 = 'that is not present among the exponent vectors of "L".'
            raise RuntimeError(msg0 + msg1 + msg2)
    C_rows = []
    for alpha_i in s_h.alpha:
        temp_func = classname.from_dict({tuple(alpha_i): 1}) * h
        c_row = relative_coeff_vector(temp_func, L.alpha)
        C_rows.append(c_row)
    C = np.vstack(C_rows)
    return C
