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
from sageopt.symbolic.signomials import Signomial
from sageopt.coniclifts import Variable, Cone, DualProductCone
import numpy as np
import itertools


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)

__EXPONENT_VECTOR_DECIMAL_POINTS__ = 7


class Elf(object):
    """
    A concrete representation for an entropy-like function of the form
    :math:`x \\mapsto f_0(x) \\sum_{i=n}^n x_i f_i(x)`
    where :math:`f_i` are signomials mapping :math:`\\mathbb{R}^n` to :math:`\\mathbb{R}`
    """

    def __init__(self, fs):
        # fs is an indexed iterable (e.g. a list) of Signomial objects or numeric types.
        # All signomials appearing in fs must be over a common number of variables,
        #   which is one less than the length of fs.
        # The first element of fs is
        if isinstance(fs, np.ndarray):
            fs = fs.tolist()
        self.n = len(fs) - 1
        self.sig = Signomial.cast(self.n, fs[0])
        self.xsigs = np.array([Signomial.cast(self.n, fi) for fi in fs[1:]])
        if any([fi.n != self.n for fi in self.xsigs]):
            raise ValueError()
        self._rmat = None
        self._smat = None

    def is_signomial(self):
        for fi in self.xsigs:
            if np.any(fi.c != 0):
                return False
        return True

    @property
    def rmat(self):
        if self._rmat is None:
            self.set_rmat()
        return self._rmat

    def set_rmat(self, **kwargs):
        # There is some freedom in what we can take as "r" vectors (since nonnegativity-certificate
        #   decompositions of these functions are not sparsity-preserving). However, we must AT
        #   LEAST consider all vectors which appear in each signomial in "xsigs".
        if len(kwargs) > 0:
            raise ValueError()
        mats = [s.alpha for s in self.xsigs]
        mat = np.vstack(mats)
        self._rmat = np.unique(mat, axis=0)

    @property
    def smat(self):
        if self._smat is None:
            self.set_smat()
        return self._smat

    def set_smat(self, **kwargs):
        # There is significant freedom in what we can consider for "s" vectors.
        #   This function says we take all vectors which appear in any constituent signomial.
        if len(kwargs) > 0:
            raise ValueError()
        mats = [self.sig.alpha]
        mats.extend([s.alpha for s in self.xsigs])
        mat = np.vstack(mats)
        self._smat = np.unique(mat, axis=0)

    def __call__(self, x, **kwargs):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        val = self.sig(x)
        val += sum([x[i] * self.xsigs[i](x) for i in range(x.size)])
        return val

    @staticmethod
    def cast(n, other):
        if isinstance(other, Elf):
            return other
        elif isinstance(other, Signomial):
            if not other.n == n:
                raise ValueError()
            fs = [other] + [0.0] * n
            f = Elf(fs)
            return f
        elif isinstance(other, __NUMERIC_TYPES__):
            fs = [Signomial.cast(n, other)] + [0.0] * n
            f = Elf(fs)
            return f
        else:
            raise NotImplementedError()

    def __add__(self, other):
        # TODO: create a quick_sum function which operates on iterables
        #   of Elfs, and calls the Signomial quick_sum when appropriate.
        other = Elf.cast(self.n, other)
        sig = self.sig + other.sig
        xsigs = self.xsigs + other.xsigs
        fs = [sig] + xsigs.tolist()
        f = Elf(fs)
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = -other
        f = self.__add__(other)
        return f

    def __rsub__(self, other):
        f = -self
        g = f + other
        return g

    def __mul__(self, other):
        try:
            other = Signomial.cast(self.n, other)
        except ValueError:
            raise ArithmeticError()
        sig = self.sig * other
        xsigs = self.xsigs * other
        fs = [sig] + xsigs.tolist()
        f = Elf(fs)
        return f

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            other = Signomial.cast(self.n, other)
        except ValueError:
            raise ArithmeticError()
        other = other ** -1
        f = self * other
        return f

    def __pow__(self, power, modulo=None):
        if not self.is_signomial():
            raise ArithmeticError()
        f = self.sig
        g = f ** power
        return g

    def __neg__(self):
        sig = -self.sig
        xsigs = -self.xsigs
        fs = [sig] + xsigs.tolist()
        f = Elf(fs)
        return f

    @staticmethod
    def sig_times_linfunc(sig, linfunc):
        # sig is a Signomial object, defined on n variables.
        # linfunc is real vector of length n.
        # return the Elf defined by f(x) = sig(x) * (linfunc @ x)
        sigx = linfunc * sig  # broadcast elementwise multiplication
        fs = [Signomial.cast(sig.n, 0)]
        fs.extend(sigx)
        f = Elf(fs)
        return f


def spelf(R, S):
    """
    Return an Elf with symbolic coefficients, obtained by taking sums of nonnegative
    functions of the form

    .. math::

        f(x) = a_r\\exp(r\\cdot x) + a_s\\exp(s\\cdot x) + a_{rs}\\exp(r\\cdot x)((r-s)\\cdot x)

    We can require nonnegativity of an individual such function by having
    :math:`(a_[s}, a_{rs}, a_{r}) \\in K_{exp}^{\\dagger}`, where :math:`K_{exp}^{\\dagger}` is
    the dual exponential cone according to the coniclifts standard.
    """
    TOL = 1e-7
    pairs = []
    for (r, s) in itertools.product(R, S):
        if np.linalg.norm(r - s) < TOL:
            continue
        pairs.append((r, s))
    num_cross_terms = len(pairs)
    c = Variable(shape=(num_cross_terms, 3), name='spelf_c')
    summand_sigs = []
    summand_elfs = []
    i = 0
    for (r, s) in pairs:
        if np.linalg.norm(r - s) < TOL:
            continue
        ci = c[i, :]
        g_r = Signomial(r.reshape((1, -1)), ci[0])
        g_s = Signomial(s.reshape((1, -1)), ci[1])
        linfunc = ci[2]*(r-s)
        g_rs = Elf.sig_times_linfunc(g_r, linfunc)
        # TODO: make a function which constructs the
        # sum of g_r, g_s, g_rs without requiring
        # signomial arithmetic (i.e. keep things fast).
        summand_sigs = [g_r, g_s]
        summand_elfs = [g_rs]
        i += 1
    K = [Cone('e', 3) for i in range(num_cross_terms)]
    c_perm = c[:, [1,2,0]]
    c_flat = c_perm.ravel(order='C')
    constr = DualProductCone(c_flat, K)
    f = Signomial.sum(summand_sigs) + sum(summand_elfs)
    return f, constr

