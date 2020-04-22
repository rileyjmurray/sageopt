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
import scipy.sparse as sp
from collections import defaultdict
from sageopt.coniclifts.constraints.elementwise import ElementwiseConstraint
from sageopt.coniclifts.constraints.set_membership.psd_cone import PSD
from sageopt.coniclifts.utilities import array_index_iterator, __REAL_TYPES__


class ScalarAtom(object):

    # A ScalarAtom is a thing that is not reduced in a ScalarExpression.

    def is_convex(self):
        return False

    def is_concave(self):
        return False

    def is_affine(self):
        return False

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return NotImplementedError()

    def value(self):
        raise NotImplementedError()

    def scalar_variables(self):
        raise NotImplementedError()


class ScalarVariable(ScalarAtom):

    _SCALAR_VARIABLE_COUNTER = 0

    @staticmethod
    def curr_variable_count():
        return ScalarVariable._SCALAR_VARIABLE_COUNTER

    def __init__(self, parent, index):
        """
        :param parent: a Variable object originally containing this ScalarVariable
        :param index: a tuple. Access the ScalarExpression containing this ScalarVariable
        with ``parent[index]``.
        """
        self._id = ScalarVariable._SCALAR_VARIABLE_COUNTER
        self._generation = parent.generation
        self._value = np.NaN
        self.index = index
        self.parent = parent
        ScalarVariable._SCALAR_VARIABLE_COUNTER += 1

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return hash((self._id, self._generation))

    def __eq__(self, other):
        if isinstance(other, ScalarVariable):
            id_okay = self._id == other._id
            gen_okay = self._generation == other._generation
            return id_okay and gen_okay
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, ScalarVariable):
            return self._id < other._id
        else:  # pragma: no cover
            raise RuntimeError('Cannot compare ScalarVariable to ' + str(type(other)))

    def __gt__(self, other):  # pragma: no cover
        if isinstance(other, ScalarVariable):
            return self._id > other._id
        else:
            raise RuntimeError('Cannot compare ScalarVariable to ' + str(type(other)))

    def is_affine(self):
        return True

    def is_convex(self):
        return True

    def is_concave(self):
        return True

    def scalar_variables(self):
        return [self]

    def value(self):
        return self._value

    def __getstate__(self):
        # We lose the link to our parent.
        # Our parent will have to restore the link later.
        d = self.__dict__.copy()
        d['parent'] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        pass


class NonlinearScalarAtom(ScalarAtom):

    @staticmethod
    def parse_arg(arg):
        if isinstance(arg, ScalarExpression) and arg.is_affine():
            arg.remove_zeros()
            res = sorted(list(arg.atoms_to_coeffs.items()))
            return tuple(res + [('OFFSET', arg.offset)])
        else:  # pragma: no cover
            raise RuntimeError('NonlinearScalarAtom arguments must be affine ScalarExpressions.')

    @staticmethod
    def __atom_text__():
        raise NotImplementedError()

    @property
    def id(self):
        # noinspection PyUnresolvedReferences
        return self._id

    @property
    def args(self):
        # noinspection PyUnresolvedReferences
        return self._args

    @property
    def evaluator(self):
        # noinspection PyUnresolvedReferences
        return self._evaluator

    @property
    def epigraph_variable(self):
        # noinspection PyUnresolvedReferences
        return self._epigraph_variable

    def __hash__(self):
        return hash(self.args + (self.__atom_text__(),))

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        else:
            return self.args == other.args

    def scalar_variables(self):
        var_list = []
        for idx in range(len(self.args)):
            var_list += [tup[0] for tup in self.args[idx][:-1]]
        return var_list

    def value(self):
        vals = []
        for arg in self.args:
            d = dict(arg[:-1])
            arg_se = ScalarExpression(d, arg[-1][1], verify=False)
            arg_val = arg_se.value
            vals.append(arg_val)
        f = self.evaluator
        res = f(vals)
        return res

    def is_convex(self):
        raise NotImplementedError()

    def is_concave(self):
        raise NotImplementedError()

    def epigraph_conic_form(self):
        """
        :return: A_vals, A_rows, A_cols, b, K
            A_vals - list (of floats)
            A_rows - numpy 1darray (of integers)
            A_cols - list (of integers)
            b - numpy 1darray (of floats)
            K - list (of coniclifts Cone objects)
        """
        raise NotImplementedError()


class ScalarExpression(object):

    __array_priority__ = 100

    def __init__(self, atoms_to_coeffs, offset, verify=True):
        """

        :param atoms_to_coeffs: a dictionary mapping ScalarAtoms to numeric types.
        :param offset: a numeric type.

        Represents sum([ c * v for (v,c) in atoms_to_coeffs.items()]) + offset.
        """
        if not isinstance(atoms_to_coeffs, defaultdict):
            self.atoms_to_coeffs = defaultdict(int)
            if verify:
                if not isinstance(offset, __REAL_TYPES__):  # pragma: no cover
                    raise RuntimeError('Coefficients in ScalarExpressions can only be numeric types.')
                if not all(isinstance(v, __REAL_TYPES__) for v in atoms_to_coeffs.values()):  # pragma: no cover
                    raise RuntimeError('Coefficients in ScalarExpressions can only be numeric types.')
                if not all(isinstance(v, ScalarAtom) for v in atoms_to_coeffs.keys()):  # pragma: no cover
                    raise RuntimeError('Keys in ScalarExpressions must be ScalarAtoms.')
            self.atoms_to_coeffs.update(atoms_to_coeffs)
        else:
            self.atoms_to_coeffs = atoms_to_coeffs.copy()
        self.offset = offset

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return self.as_expr() + other
        f = ScalarExpression(self.atoms_to_coeffs, self.offset, verify=False)
        if isinstance(other, __REAL_TYPES__):
            f.offset += other
        else:
            if isinstance(other, ScalarAtom):
                other = ScalarExpression({other: 1}, 0)
            for k, v in other.atoms_to_coeffs.items():
                f.atoms_to_coeffs[k] += v
            f.offset += other.offset
        return f

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return self.as_expr() - other
        if isinstance(other, __REAL_TYPES__):
            f = ScalarExpression(self.atoms_to_coeffs, self.offset, verify=False)
            f.offset -= other
        else:
            if isinstance(other, ScalarAtom):
                other = ScalarExpression({other: 1}, 0)
            f = ScalarExpression(self.atoms_to_coeffs, self.offset, verify=False)
            for k, v in other.atoms_to_coeffs.items():
                f.atoms_to_coeffs[k] -= v
            f.offset -= other.offset
        return f

    def __mul__(self, other):
        if isinstance(other, __REAL_TYPES__):
            if other == 0:
                return ScalarExpression(dict(), 0)
            f = ScalarExpression(self.atoms_to_coeffs, self.offset, verify=False)
            for k in f.atoms_to_coeffs:
                f.atoms_to_coeffs[k] *= other
            f.offset *= other
            return f
        elif isinstance(other, np.ndarray):
            return self.as_expr() * other
        if not isinstance(other, ScalarExpression):
            otype = str(type(other))
            raise RuntimeError('Cannot multiply ScalarExpression with object of type %s', otype)
        elif other.is_constant():
            return other.offset * self
        elif self.is_constant():
            return other * self.offset
        else:
            raise RuntimeError('Cannot multiply two non-constant ScalarExpression objects.')

    def __truediv__(self, other):
        if not isinstance(other, __REAL_TYPES__):  # pragma: no cover
            print(type(other))
            raise RuntimeError('Can only divide ScalarExpressions by scalars.')
        return self * (1 / other)

    def __neg__(self):
        f = ScalarExpression(self.atoms_to_coeffs, self.offset, verify=False)
        for k in f.atoms_to_coeffs:
            f.atoms_to_coeffs[k] *= -1
        f.offset *= -1
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        # noinspection PyTypeChecker
        return other + (-1) * self

    def remove_zeros(self):
        acs = list(self.atoms_to_coeffs.items())
        for a, c in acs:
            if c == 0:
                del self.atoms_to_coeffs[a]

    def is_constant(self):
        self.remove_zeros()
        return len(self.atoms_to_coeffs) == 0

    def is_convex(self):
        for a, c in self.atoms_to_coeffs.items():
            if a.is_affine():
                continue
            elif a.is_convex() and c < 0:
                return False
            elif a.is_concave() and c > 0:
                return False
            elif not (a.is_concave() or a.is_convex()):
                return False
        return True

    def is_concave(self):
        for a, c in self.atoms_to_coeffs.items():
            if a.is_affine():
                continue
            elif a.is_convex() and c > 0:
                return False
            elif a.is_concave() and c < 0:
                return False
            elif not (a.is_concave() or a.is_convex()):
                return False
        return True

    def is_affine(self):
        return all(isinstance(k, ScalarVariable) for k in self.atoms_to_coeffs)

    def __le__(self, other):
        # self <= other
        return ElementwiseConstraint(self, other, "<=")

    def __ge__(self, other):
        # self >= other
        return ElementwiseConstraint(self, other, ">=")

    def __eq__(self, other):
        return ElementwiseConstraint(self, other, "==")

    def scalar_variables(self):
        self.remove_zeros()
        svs = []
        for a in self.atoms_to_coeffs:
            if isinstance(a, ScalarVariable):
                svs.append(a)
            else:
                svs += a.scalar_variables()
        return list(set(svs))

    def variables(self):
        var_ids = set()
        var_list = []
        sv = self.scalar_variables()
        for v in sv:
            if id(v.parent) not in var_ids:
                var_ids.add(id(v.parent))
                var_list.append(v.parent)
        return var_list

    def as_expr(self):
        # noinspection PyTypeChecker
        return np.array(self).view(Expression)

    @property
    def value(self):
        atoms_and_coeffs = list(self.atoms_to_coeffs.items())
        atom_vals = np.array([ac[0].value() for ac in atoms_and_coeffs])
        atom_coeffs = np.array([ac[1] for ac in atoms_and_coeffs])
        val = np.dot(atom_vals, atom_coeffs) + self.offset
        return val


class Expression(np.ndarray):
    """
    An Expression is an ndarray whose entries are ScalarExpressions. Variable objects are
    a special case of Expression objects.

    Construction

        Expression objects can be constructed from ndarrays of numeric types, or ndarrays
        containing ScalarExpressions. In both cases, the construction syntax is ::

            expr = Expression(existing_array)

    Arithmetic operator overloading

        Operations ``+``, ``-``, ``/`` and ``*`` work in the exact same way as for numpy arrays.

        Expressions do not allow exponentiation (``**``).

        Expressions overload ``@`` (a.k.a. ``__matmul__``) in a way that is consistent with
        numpy, but only for arguments which have up to two dimensions.

    Constraint-based operator overloading

        Operations ``<=`` and ``>=`` produce elementwise inequality constraints.

        The operation ``==`` produces an elementwise equality constraint.

        The operations ``>>`` and ``<<`` produce linear matrix inequality constraints.

    """

    __array_priority__ = 100

    def __new__(cls, obj):
        attempt = np.array(obj, dtype=object, copy=False, subok=True)
        for tup in array_index_iterator(attempt.shape):
            # noinspection PyTypeChecker,PyCallByClass
            Expression.__setitem__(attempt, tup, attempt[tup])
        return attempt.view(Expression)

    def __matmul__(self, other):
        """
        :param other: "self" is multiplying "other" on the left.
        :return: self @ other
        """
        if other.ndim > 2 or self.ndim > 2:  # pragma: no cover
            msg = '\n \t Matmul implementation uses "dot", '
            msg += 'which behaves differently for higher dimension arrays.\n'
            raise RuntimeError(msg)
        return Expression.__rmatmul__(self.T, other.T).T

    def __rmatmul__(self, other):
        """
        :param other: a constant Expression or nd-array which left-multiplies "self".
        :return: other @ self
        """
        if other.ndim > 2 or self.ndim > 2:  # pragma: no cover
            msg = '\n \t Matmul implementation uses "dot", '
            msg += 'which behaves differently for higher dimension arrays.\n'
            raise RuntimeError(msg)
        if isinstance(other, sp.spmatrix):
            other = other.toarray()
        (A, x, B) = self.factor()
        if isinstance(other, Expression):
            if not other.is_constant():  # pragma: no cover
                raise RuntimeError('Can only multiply by constant Expressions.')
            else:
                _, _, other = other.factor()
        if other.ndim == 2:
            other_times_A = np.tensordot(other, A, axes=1)
        else:
            other_times_A = np.tensordot(other.reshape((1, -1)), A, axes=1)
            other_times_A = np.squeeze(other_times_A, axis=0)
        other_times_A_x = Expression._disjoint_dot(other_times_A, x)
        res = other_times_A_x
        other_times_B = np.tensordot(other, B, axes=1)
        for tup in array_index_iterator(other_times_A_x.shape):
            res[tup].offset = other_times_B[tup]
        return res

    def __le__(self, other):
        return ElementwiseConstraint(self, other, "<=")

    def __ge__(self, other):
        return ElementwiseConstraint(self, other, ">=")

    def __eq__(self, other):
        return ElementwiseConstraint(self, other, "==")

    def __lshift__(self, other):
        # self << other
        return PSD(other - self)

    def __rshift__(self, other):
        # self >> other
        return PSD(self - other)

    def __setitem__(self, key, value):
        if isinstance(value, ScalarExpression):
            np.ndarray.__setitem__(self, key, value)
        elif isinstance(value, __REAL_TYPES__):
            np.ndarray.__setitem__(self, key, ScalarExpression(dict(), value, verify=False))
        elif isinstance(value, ScalarAtom):
            np.ndarray.__setitem__(self, key, ScalarExpression({value: 1}, 0, verify=False))
        elif isinstance(value, np.ndarray) and value.size == 1:
            # noinspection PyTypeChecker
            self[key] = np.asscalar(value)
        elif isinstance(value, Expression):
            np.ndarray.__setitem__(self, key, value)
        else:  # pragma: no cover
            raise RuntimeError('Can only initialize with numeric, ScalarExpression, or ScalarAtom types.')

    def scalar_atoms(self):
        """
        Return a list of all ScalarAtoms appearing in this Expression.
        """
        return list(set.union(*[set(se.scalar_atoms()) for se in self.flat]))

    def scalar_variables(self):
        """
        Return a list of all ScalarVariables appearing in this Expression.
        """
        return list(set.union(*[set(se.scalar_variables()) for se in self.flat]))

    def variables(self):
        """
        Return a list of all Variable objects appearing in this Expression.
        You can assume that all Variable objects will be "proper".
        """
        var_ids = set()
        var_list = []
        for se in self.ravel():
            for sv in se.scalar_variables():
                if id(sv.parent) not in var_ids:
                    var_ids.add(id(sv.parent))
                    var_list.append(sv.parent)
        return var_list

    def is_affine(self):
        """
        True if the Expression is an affine function of Variables within its scope.
        """
        if not Expression._can_assume_affine(self):
            return all(v.is_affine() for v in self.flat)
        else:
            return True

    def is_constant(self):
        """
        True if no Variables appear in this Expression.
        """
        return all(v.is_constant() for v in self.flat)

    def is_convex(self):
        """
        Return an ndarray of booleans. For a fixed component index, the value
        of the returned array indicates if that component of the current Expression
        is a convex function of Variables within its scope.
        """
        res = np.empty(shape=self.shape, dtype=bool)
        for tup in array_index_iterator(self.shape):
            res[tup] = self[tup].is_convex()
        return res

    def is_concave(self):
        """
        Return an ndarray of booleans. For a fixed component index, the value
        of the returned array indicates if that component of the current Expression
        is a concave function of Variables within its scope.
        """
        res = np.empty(shape=self.shape, dtype=bool)
        for tup in array_index_iterator(self.shape):
            res[tup] = self[tup].is_concave()
        return res

    def as_expr(self):
        """
        Return self.
        """
        return self

    def factor(self):
        """
        Returns a tuple ``(A, x, B)``.

        ``A`` is a tensor of one order higher than the current Expression object, i.e.
        ``A.ndim == self.ndim + 1``. The dimensions of ``A`` and ``self`` agree up until
        ``self.ndim``, i.e. ``A.shape[:-1] == self.shape``.

        ``x`` is a list of ScalarAtom objects, with ``len(x) == A.shape[-1]``.

        ``B`` is a numpy array of the same shape as ``self``.

        The purpose of this function is to enable faster matrix multiplications of Expression
        objects. The idea is that if you tensor-contract ``A`` along its final dimension according
        to ``x``, and then add ``B``, you recover this Expression.
        """
        x = list(set(a for se in self.flat for a in se.atoms_to_coeffs))
        x.sort(key=lambda a: a.id)
        # Sorting by ScalarAtom id makes this method deterministic
        # when all ScalarAtoms in this Expression are of the same type.
        # That's useful for, e.g. affine Expressions, which
        # we need to test for symbolic equivalence.
        atoms_to_pos = {a: i for (i, a) in enumerate(x)}
        A = np.zeros(self.shape + (len(x),))
        B = np.zeros(self.shape)
        for tup in array_index_iterator(self.shape):
            se = self[tup]
            for a, c in se.atoms_to_coeffs.items():
                A[tup + (atoms_to_pos[a],)] = c
            B[tup] = se.offset
        return A, x, B

    @property
    def value(self):
        """
        An ndarray containing numeric entries, of shape equal to ``self.shape``.
        This is the result of propagating the value of ScalarVariable objects
        through the symbolic operations tracked by this Expression.
        """
        val = np.zeros(shape=self.shape)
        for tup in array_index_iterator(self.shape):
            val[tup] = self[tup].value
        return val

    @staticmethod
    def _disjoint_dot(array, list_of_atoms):
        # This is still MUCH SLOWER than adding numbers together.
        if len(list_of_atoms) != array.shape[-1]:  # pragma: no cover
            raise RuntimeError('Incompatible dimensions to disjoint_dot.')
        expr = np.empty(shape=array.shape[:-1], dtype=object)
        for tup in array_index_iterator(expr.shape):
            dict_items = []
            for i, a in enumerate(list_of_atoms):
                dict_items.append((a, array[tup + (i,)]))
            d = dict(dict_items)
            expr[tup] = ScalarExpression(d, 0, verify=False)
        return expr.view(Expression)

    @staticmethod
    def _can_assume_affine(expr):
        unadorned_var = isinstance(expr, (Variable, ScalarVariable))
        constant_array = isinstance(expr, np.ndarray) and expr.dtype != object
        numeric = isinstance(expr, __REAL_TYPES__)
        return unadorned_var or constant_array or numeric

    @staticmethod
    def are_equivalent(expr1, expr2, rtol=1e-5, atol=1e-8):
        """
        Perform a check that ``expr1`` and ``expr2`` are symbolically equivalent, in the
        sense of affine operators applied to ScalarAtoms. The equivalence is up to numerical
        tolerance in the sense of ``np.allclose``.

        Parameters
        ----------
        expr1 : Expression
        expr2 : Expression
        rtol : float
            relative numerical tolerance
        atol : float
            absolute numerical tolerance

        Returns
        -------
        True if the Expressions can be verified as symbolically equivalent. False otherwise.

        Notes
        -----
        The behavior of this function is conservative. If ``self`` contains a mix of
        ScalarAtoms (e.g. ScalarVariables and NonlinearScalarAtoms), then this function
        might return False even when ``expr1`` and ``expr2`` are equivalent. This is
        due to nondeterministic behavior of ``Expression.factor`` in such situations.

        """
        if not isinstance(expr1, Expression):
            expr1 = Expression(expr1)
        if not isinstance(expr2, Expression):
            expr2 = Expression(expr2)
        if expr1.shape != expr2.shape:
            return False
        A1, x1, B1 = expr1.factor()
        A2, x2, B2 = expr2.factor()
        for i in range(len(x1)):
            if not isinstance(x2[i], type(x1[i])):
                return False
            elif x1[i].id != x2[i].id:
                return False
        if not np.allclose(A1, A2, rtol=rtol, atol=atol):
            return False
        elif not np.allclose(B1, B2, rtol=rtol, atol=atol):
            return False
        else:
            return True


class Variable(Expression):
    """
    An abstraction for a symbol appearing in constraint sets, or optimization problems.

    Variable objects are a custom subclass of numpy ndarrays.

    Parameters
    ----------
    shape : tuple

        The dimensions of the Variable object. Defaults to ``shape=()``.

    name : str

        A string which should uniquely identify this Variable object in all models
        where it appears. Ideally, this string should be human-readable.
        Defaults to ``'unnamed_var_N'``,  where ``N`` is an integer.

    var_properties : list of str

        Currently, the only accepted forms of this argument are the empty
        list (in which case an unstructured Variable is returned), or a list
        containing the string ``'symmetric'`` (in which case a symmetric matrix
        Variable is returned).

    Examples
    --------

    The symbol you use in the Python interpreter does not need to match the "name" of a Variable. ::

        x = Variable(shape=(3,), name='my_name')

    A Variable object can take on any dimension that a numpy ndarray could take on. ::

        y = Variable(shape=(10,4,1,2), name='strange_shaped_var')


    Notes
    -----

    Upon construction, Variable objects are "proper". If you index into them, they are
    still considered Variable objects, but they no longer contain information about
    all of their components. A Variable object's ``name`` field only uniquely determines
    the "proper" version of that Variable. If ``v.is_proper() == False``, then it should
    be possible to recover the original Variable object with ``original_v = v.base``. ::

        x = Variable(shape=(3,), name='x')
        print(type(x))  # sageopt.coniclifts.base.Variable
        print(x.is_proper())  # True

        y = x[1:]
        print(type(y))  # sageopt.coniclifts.base.Variable
        print(y.is_proper())  # False
        print(x.name == y.name)  # True
        print(id(x) == id(y.base))  # True; these exist at the same place in memory.

    """

    _UNNAMED_VARIABLE_CALL_COUNT = 0

    _VARIABLE_GENERATION = 0

    # noinspection PyInitNewSignature
    def __new__(cls, shape=(), name=None, var_properties=None):
        if var_properties is None:
            var_properties = []
        if name is None:
            name = 'unnamed_var_' + str(Variable._UNNAMED_VARIABLE_CALL_COUNT)
            Variable._UNNAMED_VARIABLE_CALL_COUNT += 1
        obj = np.empty(shape=shape, dtype=object).view(Variable)
        obj._is_proper = True
        obj._name = name
        obj._var_properties = var_properties
        obj._scalar_variable_ids = []
        obj._generation = Variable._VARIABLE_GENERATION
        if len(var_properties) == 0:
            Variable.__unstructured_populate__(obj)
        elif 'symmetric' in var_properties:
            Variable.__symmetric_populate__(obj)
        else:
            Variable.__unstructured_populate__(obj)
            raise UserWarning('The variable with name ' + name + ' was declared with an unknown property.')
        if obj.size == 0:  # pragma: no cover
            raise RuntimeError('Cannot declare Variables with zero components.')
        if obj._scalar_variable_ids[-1] > np.iinfo(np.int).max:  # pragma: no cover
            # ScalarVariable objects can no longer be properly tracked
            msg = 'An index used by coniclifts\' backend has overflowed. \n'
            msg += 'Call coniclifts.clear_variable_indices(), and build your model again.'
            raise RuntimeError(msg)
        return obj

    # noinspection PyProtectedMember
    @staticmethod
    def __unstructured_populate__(obj):
        if obj.shape == ():
            v = ScalarVariable(parent=obj, index=tuple())
            np.ndarray.__setitem__(obj, tuple(), ScalarExpression({v: 1}, 0, verify=False))
            obj._scalar_variable_ids.append(v.id)
        else:
            for tup in array_index_iterator(obj.shape):
                v = ScalarVariable(parent=obj, index=tup)
                obj._scalar_variable_ids.append(v.id)
                np.ndarray.__setitem__(obj, tup, ScalarExpression({v: 1}, 0, verify=False))
        pass

    # noinspection PyProtectedMember
    @staticmethod
    def __symmetric_populate__(obj):
        if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:  # pragma: no cover
            raise RuntimeError('Symmetric variables must be 2d, and square.')
        temp_id_array = np.zeros(shape=obj.shape, dtype=int)
        for i in range(obj.shape[0]):
            v = ScalarVariable(parent=obj, index=(i, i))
            np.ndarray.__setitem__(obj, (i, i), ScalarExpression({v: 1}, 0, verify=False))
            temp_id_array[i, i] = v.id
            for j in range(i+1, obj.shape[1]):
                v = ScalarVariable(parent=obj, index=(i, j))
                np.ndarray.__setitem__(obj, (i, j), ScalarExpression({v: 1}, 0, verify=False))
                np.ndarray.__setitem__(obj, (j, i), ScalarExpression({v: 1}, 0, verify=False))
                temp_id_array[i, j] = v.id
                temp_id_array[j, i] = v.id
        for tup in array_index_iterator(obj.shape):
            obj._scalar_variable_ids.append(temp_id_array[tup])
        pass

    def __setitem__(self, key, value):  # pragma: no cover
        raise RuntimeError('Cannot reassign entries of Variable objects.')

    def __add__(self, other):
        return Expression.__add__(self.view(Expression), other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return Expression.__mul__(self.view(Expression), other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Expression.__neg__(self.view(Expression))

    def __sub__(self, other):
        return Expression.__sub__(self.view(Expression), other)

    def __rsub__(self, other):
        return other + self.__neg__()

    def __rmatmul__(self, other):
        return Expression.__rmatmul__(self.view(Expression), other)

    def __reduce__(self):
        # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(Variable, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        var_props = getattr(self, '_var_properties', [])
        extra_info = (self.is_proper(), self.name, var_props,
                      self.scalar_variable_ids, self.generation)
        new_state = pickled_state[2] + extra_info
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        desired_reduce = (pickled_state[0], pickled_state[1], new_state)
        return desired_reduce

    # noinspection PyMethodOverriding
    def __setstate__(self, state):
        # set the custom attributes
        self._generation = state[-1]
        self._scalar_variable_ids = state[-2]
        self._var_properties = state[-3]
        self._name = state[-4]
        self._is_proper = state[-5]
        # fall back on the
        np.ndarray.__setstate__(self, state[:-5])
        self._relink_scalar_variables()

    def is_constant(self):
        return False

    def is_affine(self):
        return True

    def scalar_variables(self):
        #TODO: make this faster for "proper" Variable objects.
        return [list(se.atoms_to_coeffs)[0] for se in self.flat]

    def leading_scalar_variable_id(self):
        if self.is_proper():
            return self._scalar_variable_ids[0]
        else:
            return self[(0,) * len(self.shape)].scalar_variables()[0].id

    @property
    def scalar_variable_ids(self):
        """
        Each component of this Variable object (i.e. each "scalar variable") contains
        an index which uniquely identifies it in all models where this Variable appears.
        Return the list of these indices.
        """
        if self.is_proper():
            return self._scalar_variable_ids
        else:
            return [self[tup].scalar_variables()[0].id for tup in array_index_iterator(self.shape)]

    @property
    def name(self):
        """
        A string which should uniquely identify this object in all models
        where it appears, provided ``self.is_proper() == True``.
        """
        if hasattr(self, '_name'):
            return self._name
        else:
            tup = (0,) * len(self.shape)
            prop_var = self[tup].scalar_variables()[0].parent
            name = prop_var.name
            return name

    @property
    def generation(self):
        """
        An internally-maintained index. Variable objects of different "generation" cannot
        participate in a common optimization problem.
        """
        if hasattr(self, '_generation'):
            return self._generation
        else:
            sv = self[(0,) * len(self.shape)].scalar_variables()[0]
            gen = sv.parent.generation
            return gen

    @property
    def value(self):
        """
        An ndarray containing numeric entries, of shape equal to ``self.shape``.
        This is the result of the most recent call to ``set_scalar_variables``.
        """
        expr = self.view(Expression)
        val = expr.value
        return val

    @value.setter
    def value(self, value):
        if isinstance(value, __REAL_TYPES__):
            value = np.array(value)
        if value.shape != self.shape:  # pragma: no cover
            raise RuntimeError('Dimension mismatch.')
        for tup in array_index_iterator(self.shape):
            sv = list(self[tup].atoms_to_coeffs)[0]
            sv._value = value[tup]
        pass

    def is_proper(self):
        return getattr(self, '_is_proper', False)

    def _relink_scalar_variables(self):
        if not self.is_proper():
            pass
        svs = self.scalar_variables()
        for sv in svs:
            sv.parent = self

