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
import sageopt.coniclifts as cl
from sageopt.symbolic import utilities as sym_util
from sageopt.symbolic.signomials import Signomial
import numpy as np
import copy
import warnings


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)

__EXPONENT_VECTOR_DECIMAL_POINTS__ = 7


class Lenomial(object):
    """
    A concrete representation for a function of the form
    :math:`x \\mapsto f_0(x) \\sum_{i=n}^n x_i f_i(x)`
    where :math:`f_i` are signomials mapping :math:`\\mathbb{R}^n` to :math:`\\mathbb{R}`

    The name for this class is not that great. The "le" is supposed
    to stand for "log entropy". Normal entropy is :math:`p \\log p``.
    Instead of :math:`p`, our variables are :math:`x = \\log p``.
    The intended pronunciation of this class is to say "len" and then "omial"
    with no pause in between.
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
        if isinstance(other, Lenomial):
            return other
        elif isinstance(other, Signomial):
            if not other.n == n:
                raise ValueError()
            fs = [other] + [0.0] * n
            f = Lenomial(fs)
            return f
        elif isinstance(other, __NUMERIC_TYPES__):
            fs = [Signomial.cast(n, other)] + [0.0] * n
            f = Lenomial(fs)
            return f
        else:
            raise NotImplementedError()

    def __add__(self, other):
        other = Lenomial.cast(self.n, other)
        sig = self.sig + other.sig
        xsigs = self.xsigs + other.xsigs
        fs = [sig] + xsigs.tolist()
        f = Lenomial(fs)
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
        f = Lenomial(fs)
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
        f = Lenomial(fs)
        return f

    @staticmethod
    def sig_times_linfunc(sig, linfunc):
        # sig is a Signomial object, defined on n variables.
        # linfunc is real vector of length n.
        # return the Lenomial defined by f(x) = sig(x) * (linfunc @ x)
        sigx = linfunc * sig  # broadcast elementwise multiplication
        fs = [Signomial.cast(sig.n, 0)]
        fs.extend(sigx)
        f = Lenomial(fs)
        return f
