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
from sageopt.symbolic.signomials import Signomial, standard_sig_monomials
from sageopt.coniclifts import Variable, Cone, DualProductCone
from sageopt.coniclifts.base import ScalarExpression, Expression
from sageopt.coniclifts.operators import affine as aff
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
        self._exp_grad = None

    def is_signomial(self):
        for fi in self.xsigs:
            if isinstance(fi.c, Expression):
                if not fi.c.is_constant():
                    return False
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

    @property
    def exp_grad(self):
        self._cache_exp_grad()
        return self._exp_grad

    def _cache_exp_grad(self):
        y = standard_sig_monomials(self.n)
        self.sig._cache_exp_grad()
        for g in self.xsigs:
            g._cache_exp_grad()
        self._exp_grad = np.empty(shape=(self.n, ), dtype=object)
        for i in range(self.n):
            self._exp_grad[i] = self._exp_partial(i, y[i])
        pass

    def _exp_partial(self, i, yi):
        eye = np.eye(self.n)
        summands = [self.sig.exp_grad[i]]
        for j in range(self.n):
            summand = Elf.sig_times_linfunc(self.xsigs[j].exp_grad[i], eye[j, :])
            if i == j:
                # do something
                summand = summand + self.xsigs[j]/yi
            summands.append(summand)
        g = Elf.sum(summands)
        return g

    def __call__(self, x, **kwargs):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        val = self.sig(x)
        val += sum([x[i] * self.xsigs[i](x) for i in range(x.size)])
        return val

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

    def fix_coefficients(self):
        sig = self.sig.fix_coefficients()
        xsigs = [fi.fix_coefficients() for fi in self.xsigs]
        f = Elf([sig] + xsigs)
        return f

    def without_zeros(self, tol=0.0):
        sig = self.sig.without_zeros(tol)
        xsigs = [fi.without_zeros(tol) for fi in self.xsigs]
        f = Elf([sig] + xsigs)
        return f

    @staticmethod
    def sum(funcs):
        n = 0
        for f in funcs:
            if hasattr(f, 'n'):
                n = f.n
                break
        if n == 0:
            raise ValueError()
        funcs = [Elf.cast(n, f) for f in funcs]
        sig = Signomial.sum([f.sig for f in funcs])
        all_sigs = [sig]
        for i in range(n):
            xsig = Signomial.sum([f.xsigs[i] for f in funcs])
            all_sigs.append(xsig)
        f = Elf(all_sigs)
        return f

    @staticmethod
    def sig_times_linfunc(sig, linfunc):
        # sig is a Signomial object, defined on n variables.
        # linfunc is real vector of length n.
        # return the Elf defined by f(x) = sig(x) * (linfunc @ x)
        linfunc = linfunc.ravel()
        fs = [Signomial.cast(sig.n, 0)]
        for i, val in enumerate(linfunc):
            if val == 0:
                sigi = Signomial.cast(sig.n, 0)
            else:
                sigi = Signomial(sig.alpha, sig.c * val)
            fs.append(sigi)
        f = Elf(fs)
        return f

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
        elif isinstance(other, __NUMERIC_TYPES__) or\
                (hasattr(other, 'size') and other.size == 1) or\
                isinstance(other, ScalarExpression):
            fs = [Signomial.cast(n, other)] + [0.0] * n
            f = Elf(fs)
            return f
        else:
            raise NotImplementedError()


def spelf(R, S, zero_origin=True, name='spelf_c'):
    """
    Return an Elf with symbolic coefficients, obtained by taking sums of nonnegative
    functions of the form

    .. math::

        f(x) = a_r\\exp(r\\cdot x) + a_s\\exp(s\\cdot x) + a_{rs}\\exp(r\\cdot x)((r-s)\\cdot x)

    We can require nonnegativity of an individual such function by having
    :math:`(-a_{rs}, a_s, a_r) \\in K_{exp}^{\\dagger}`
    where :math:`K_{exp}^{\\dagger}` is the dual exponential cone using the coniclifts standard.
    """
    TOL = 1e-7
    pairs = []
    for (r, s) in itertools.product(R, S):
        if np.linalg.norm(r - s) < TOL:
            continue
        pairs.append((r, s))
    num_cross_terms = len(pairs)
    summand_sigs = []
    summand_elfs = []
    summand_melfs = []
    i = 0
    if zero_origin:
        c = Variable(shape=(num_cross_terms,), name=name)
        for (r, s) in pairs:
            r, s = r.reshape((1, -1)), s.reshape((1, -1))
            if np.linalg.norm(r - s) < TOL:
                continue
            g_r = Signomial(r, -c[i])
            g_s = Signomial(s, c[i])
            g_rs = Elf.sig_times_linfunc(Signomial(r, c[i]), r-s)
            summand_sigs += [g_r, g_s]
            summand_elfs += [g_rs]
            summand_melfs.append((g_r, g_s, g_rs))
            i += 1
        constr = c >= 0
    else:
        c = Variable(shape=(num_cross_terms, 3), name=name)
        for (r, s) in pairs:
            r, s = r.reshape((1, -1)), s.reshape((1, -1))
            if np.linalg.norm(r - s) < TOL:
                continue
            ci = c[i, :]
            g_r = Signomial(r, ci[0])
            g_s = Signomial(s, ci[1])
            g_rs = Elf.sig_times_linfunc(Signomial(r, ci[2]), r-s)
            # TODO: make a function which constructs the
            #   sum of g_r, g_s, g_rs without requiring
            #   signomial arithmetic (i.e. keep things fast).
            summand_sigs += [g_r, g_s]
            summand_elfs += [g_rs]
            summand_melfs.append((g_r, g_s, g_rs))
            i += 1
        K = [Cone('e', 3) for i in range(num_cross_terms)]
        c_mod = aff.column_stack((-c[:, 2], c[:, 1], c[:, 0]))
        c_flat = c_mod.ravel(order='C')
        constr = DualProductCone(c_flat, K)
    f = Signomial.sum(summand_sigs) + Elf.sum(summand_elfs)
    return f, constr, summand_melfs

