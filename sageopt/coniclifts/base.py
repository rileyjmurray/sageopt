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

    def __add__(self, other):
        return ScalarExpression({self: 1}, 0) + other

    def __sub__(self, other):
        return ScalarExpression({self: 1}, 0) - other

    def __mul__(self, other):
        if not isinstance(other, __REAL_TYPES__):
            raise RuntimeError('Can only multiply by scalars.')
        return ScalarExpression({self: other}, 0)

    def __truediv__(self, other):
        if not isinstance(other, __REAL_TYPES__):
            raise RuntimeError('Can only divide by scalars.')
        return ScalarExpression({self: (1 / other)}, 0)

    def __neg__(self):
        return ScalarExpression({self: -1}, 0)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return other + ScalarExpression({self: -1}, 0)

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

    def __init__(self, parent, name):
        """
        :param parent: a Variable object originally containing this ScalarVariable
        :param name: A string; likely the parent's name followed by a subscript.
        """
        self._id = ScalarVariable._SCALAR_VARIABLE_COUNTER
        self._generation = parent.generation
        self._value = np.NaN
        self.name = name
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
        else:
            raise RuntimeError('Cannot compare ScalarVariable to ' + str(type(other)))

    def __gt__(self, other):
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
        # Type conversion
        if isinstance(arg, ScalarVariable):
            arg = ScalarExpression({arg: 1}, 0)
        elif isinstance(arg, Expression) and arg.size == 1:
            arg = np.asscalar(arg)
            # noinspection PyTypeChecker
            return NonlinearScalarAtom.parse_arg(arg)
        elif isinstance(arg, __REAL_TYPES__):
            arg = ScalarExpression(dict(), arg)
        # Parsing
        if isinstance(arg, ScalarExpression) and arg.is_affine():
            arg.remove_zeros()
            res = sorted(list(arg.atoms_to_coeffs.items()))
            return tuple(res + [('OFFSET', arg.offset)])
        else:
            raise RuntimeError('ScalarAtom arguments must be affine ScalarExpressions.')

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
        f = self.evaluator
        # construct numeric values for argument expressions
        args_val = np.zeros(shape=(len(self.args),))
        for idx, arg in enumerate(self.args):
            var_vals = np.array([tup[0].value() for tup in arg[:-1]])
            var_coeffs = np.array([tup[1] for tup in arg[:-1]])
            args_val[idx] = np.dot(var_coeffs, var_vals) + arg[-1][1]
        # evaluate the defining function, and return the result
        val = f(args_val)
        return val

    def is_convex(self):
        raise NotImplementedError()

    def is_concave(self):
        raise NotImplementedError()

    def epigraph_conic_form(self):
        """
        :return: A_vals, A_rows, A_cols, b, K, aux_var, sep_K
            A_vals - list (of floats)
            A_rows - numpy 1darray (of integers)
            A_cols - list (of integers)
            b - numpy 1darray (of floats)
            K - list (of coniclifts Cone objects)
            aux_var - a ScalarVariable object for the epigraph of this ScalarAtom
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
                vals = list(atoms_to_coeffs.values()) + [offset]
                if not all(isinstance(v, __REAL_TYPES__) for v in vals):
                    raise RuntimeError('Coefficients in ScalarExpressions can only be numeric types.')
                keys = list(atoms_to_coeffs.keys())
                if not all(isinstance(v, ScalarAtom) for v in keys):
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
        elif isinstance(other, ScalarExpression) and other.is_constant():
            return other.offset * self
        else:
            return other.__rmul__(self)

    def __truediv__(self, other):
        if not isinstance(other, __REAL_TYPES__):
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

    def value(self):
        atoms_and_coeffs = list(self.atoms_to_coeffs.items())
        atom_vals = np.array([ac[0].value() for ac in atoms_and_coeffs])
        atom_coeffs = np.array([ac[1] for ac in atoms_and_coeffs])
        val = np.dot(atom_vals, atom_coeffs) + self.offset
        return val


class Expression(np.ndarray):

    """
    SUBCLASSING NDARRAY

    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
    """

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
        if other.ndim > 2 or self.ndim > 2:
            msg = '\n \t Matmul implementation uses "dot", '
            msg += 'which behaves differently for higher dimension arrays.\n'
            raise RuntimeError(msg)
        return Expression.__rmatmul__(self.T, other.T).T

    def __rmatmul__(self, other):
        """
        :param other: a constant Expression or nd-array which left-multiplies "self".
        :return: other @ self
        """
        if other.ndim > 2 or self.ndim > 2:
            msg = '\n \t Matmul implementation uses "dot", '
            msg += 'which behaves differently for higher dimension arrays.\n'
            raise RuntimeError(msg)
        (A, x, B) = self.factor()
        if isinstance(other, Expression):
            if not other.is_constant():
                raise RuntimeError('Can only multiply by constant Expressions.')
            else:
                _, _, other = other.factor()
        if other.ndim == 2:
            other_times_A = np.tensordot(other, A, axes=1)
        else:
            other_times_A = np.tensordot(other.reshape((1, -1)), A, axes=1)
            other_times_A = np.squeeze(other_times_A, axis=0)
        other_times_A_x = Expression.disjoint_dot(other_times_A, x)
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
        else:
            raise RuntimeError('Can only initialize with numeric, ScalarExpression, or ScalarAtom types.')

    def scalar_atoms(self):
        return list(set.union(*[set(se.scalar_atoms()) for se in self.flat]))

    def scalar_variables(self):
        return list(set.union(*[set(se.scalar_variables()) for se in self.flat]))

    def variables(self):
        var_ids = set()
        var_list = []
        for se in self.ravel():
            for sv in se.scalar_variables():
                if id(sv.parent) not in var_ids:
                    var_ids.add(id(sv.parent))
                    var_list.append(sv.parent)
        return var_list

    def is_affine(self):
        if not Expression.can_assume_affine(self):
            return all(v.is_affine() for v in self.flat)
        else:
            return True

    def is_constant(self):
        return all(v.is_constant() for v in self.flat)

    def is_convex(self):
        res = np.empty(shape=self.shape, dtype=bool)
        for tup in array_index_iterator(self.shape):
            res[tup] = self[tup].is_convex()
        return res

    def is_concave(self):
        res = np.empty(shape=self.shape, dtype=bool)
        for tup in array_index_iterator(self.shape):
            res[tup] = self[tup].is_concave()
        return res

    def as_expr(self):
        return self

    def factor(self):
        """
        :return: (A, x, b) -- A is a numpy ndarray of numeric scalars. x is an Expression object
        where each entry in x.ravel() is a ScalarExpression consisting of a single atom
        (with zero offset), and b is a numpy array of same dimensions as the Expression
        np.matmul(A, x).
        """
        x = list(set(a for se in self.flat for a in se.atoms_to_coeffs))
        x.sort(key=lambda a: a.id)
        # Sorting by ScalarAtom id makes this method deterministic
        # when all ScalarAtoms in this Expression are of the same type.
        # That's useful for, e.g. affine Expressions, which
        # we need to test for symbolic equivalence.
        atoms_to_pos = dict((a, i) for (i, a) in enumerate(x))
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
        val = np.zeros(shape=self.shape)
        for tup in array_index_iterator(self.shape):
            val[tup] = self[tup].value()
        return val

    @staticmethod
    def disjoint_dot(array, list_of_atoms):
        # This is still MUCH SLOWER than adding numbers together.
        if len(list_of_atoms) != array.shape[-1]:
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
    def disjoint_add(expr1, expr2):
        if expr1.shape != expr2.shape:
            raise RuntimeError('Disjoint add requires operands of the same shape.')
        expr3 = np.empty(expr1.shape, dtype=object)
        for tup in array_index_iterator(expr3.shape):
            d = dict(expr1[tup].atoms_to_coeffs)
            d.update(expr2[tup].atoms_to_coeffs)
            offset = expr1[tup].offset + expr2[tup].offset
            expr3[tup] = ScalarExpression(d, offset)
        return expr3.view(Expression)

    @staticmethod
    def can_assume_affine(expr):
        unadorned_var = isinstance(expr, (Variable, ScalarVariable))
        constant_array = isinstance(expr, np.ndarray) and expr.dtype != object
        numeric = isinstance(expr, __REAL_TYPES__)
        return unadorned_var or constant_array or numeric

    @staticmethod
    def are_equivalent(expr1, expr2, rtol=1e-5, atol=1e-8):
        # WARNING:
        #   The behavior of this function is conservative.
        #   Non-deterministic behavior of factor(...) when either Expression
        #   contains a mix of different ScalarAtoms (e.g. ScalarVariables
        #   and NonlinearScalarAtoms) can cause this function to return
        #   False when the expressions are in fact equivalent.
        #
        #   This function only works on 0d, 1d, and 2d expressions.
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
            Variable.__unstructured_populate__(obj, name)
        elif 'symmetric' in var_properties:
            Variable.__symmetric_populate__(obj, name)
        else:
            Variable.__unstructured_populate__(obj, name)
            raise UserWarning('The variable with name ' + name + ' was declared with an unknown property.')
        return obj

    # noinspection PyProtectedMember
    @staticmethod
    def __unstructured_populate__(obj, name):
        if obj.shape == ():
            v = ScalarVariable(parent=obj, name=name)
            np.ndarray.__setitem__(obj, tuple(), ScalarExpression({v: 1}, 0, verify=False))
            obj._scalar_variable_ids.append(v.id)
        else:
            for tup in array_index_iterator(obj.shape):
                v = ScalarVariable(parent=obj, name=name + str(list(tup)))
                obj._scalar_variable_ids.append(v.id)
                np.ndarray.__setitem__(obj, tup, ScalarExpression({v: 1}, 0, verify=False))
        pass

    # noinspection PyProtectedMember
    @staticmethod
    def __symmetric_populate__(obj, name):
        if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:
            raise RuntimeError('Symmetric variables must be 2d, and square.')
        temp_id_array = np.zeros(shape=obj.shape, dtype=int)
        for i in range(obj.shape[0]):
            v = ScalarVariable(parent=obj, name=name + str([i, i]))
            np.ndarray.__setitem__(obj, (i, i), ScalarExpression({v: 1}, 0, verify=False))
            temp_id_array[i, i] = v.id
            for j in range(i+1, obj.shape[1]):
                v = ScalarVariable(parent=obj, name=name + str([i, j]))
                np.ndarray.__setitem__(obj, (i, j), ScalarExpression({v: 1}, 0, verify=False))
                np.ndarray.__setitem__(obj, (j, i), ScalarExpression({v: 1}, 0, verify=False))
                temp_id_array[i, j] = v.id
                temp_id_array[j, i] = v.id
        for tup in array_index_iterator(obj.shape):
            obj._scalar_variable_ids.append(temp_id_array[tup])
        pass

    def __setitem__(self, key, value):
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

    # noinspection PyMethodOverriding
    def __setstate__(self, state):
        np.ndarray.__setstate__(self, state)
        self.relink_scalar_variables()

    def is_constant(self):
        return False

    def is_affine(self):
        return True

    def scalar_variables(self):
        return [list(se.atoms_to_coeffs)[0] for se in self.flat]

    def leading_scalar_variable_id(self):
        if self.is_proper():
            return self._scalar_variable_ids[0]
        else:
            return self[(0,) * len(self.shape)].scalar_variables()[0].id

    def set_scalar_variables(self, value):
        if value.shape != self.shape:
            raise RuntimeError('Dimension mismatch.')
        for tup in array_index_iterator(self.shape):
            sv = list(self[tup].atoms_to_coeffs)[0]
            sv._value = value[tup]
        pass

    @property
    def scalar_variable_ids(self):
        if self.is_proper():
            return self._scalar_variable_ids
        else:
            return [self[tup].scalar_variables()[0].id for tup in array_index_iterator(self.shape)]

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        else:
            tup = (0,) * len(self.shape)
            temp = self[(0,) * len(self.shape)].scalar_variables()[0].name
            if self.ndim > 0:
                # Need to trim the indices at the end of the string
                return temp[:-len(str(list(tup)))]
            else:
                return temp

    @property
    def generation(self):
        return self._generation

    def is_proper(self):
        return hasattr(self, '_is_proper')

    def relink_scalar_variables(self):
        if not self.is_proper():
            pass
        svs = self.scalar_variables()
        for sv in svs:
            sv.parent = self

