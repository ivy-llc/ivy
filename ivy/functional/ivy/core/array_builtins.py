"""
Collection of builtin Ivy functions.
"""

# global
from typing import Union

# local
import ivy


def builtin_dir(x):
    return x.__dir__()


# noinspection PyShadowingBuiltins
def builtin_getattr(x, item):
    return x.__getattr__(item)


# noinspection PyShadowingBuiltins
def builtin_getattribute(x, item):
    return x.__getattribute__(item)


# noinspection PyShadowingBuiltins
def builtin_getitem(x, query):
    return x.__getitem__(query)


# noinspection PyShadowingBuiltins
def builtin_setitem(x, query, val):
    return x.__setitem__(query, val)


# noinspection PyShadowingBuiltins
def builtin_contains(x, key):
    return x.__contains__(key)


# noinspection PyShadowingBuiltins
def builtin_pos(x):
    return x.__pos__()


# noinspection PyShadowingBuiltins
def builtin_neg(x):
    return x.__neg__()


# noinspection PyShadowingBuiltins
def builtin_pow(self: Union[ivy.Array, ivy.NativeArray],
                other: Union[int, float, ivy.Array, ivy.NativeArray]) \
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of an
    array instance to the power of other_i (the exponent), where other_i is the corresponding element of the array other.

    :param self: array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
    :param other: other array whose elements correspond to the exponentiation exponent. Must be compatible with x
                    (see Broadcasting). Should have a numeric data type.
    :return: an array containing the element-wise results. The returned array must have a data type determined by
              Type Promotion Rules.
    """
    return self.__pow__(other)


# noinspection PyShadowingBuiltins
def builtin_rpow(x, power):
    return x.__rpow__(power)


# noinspection PyShadowingBuiltins
def builtin_add(x, other):
    return x.__add__(other)


# noinspection PyShadowingBuiltins
def builtin_radd(x, other):
    return x.__radd__(other)


# noinspection PyShadowingBuiltins
def builtin_sub(x, other):
    return x.__sub__(other)


# noinspection PyShadowingBuiltins
def builtin_rsub(x, other):
    return x.__rsub__(other)


# noinspection PyShadowingBuiltins
def builtin_mul(x, other):
    return x.__mul__(other)


# noinspection PyShadowingBuiltins
def builtin_rmul(x, other):
    return x.__rmul__(other)


# noinspection PyShadowingBuiltins
def builtin_truediv(x, other):
    return x.__truediv__(other)


# noinspection PyShadowingBuiltins
def builtin_rtruediv(x, other):
    return x.__rtruediv__(other)


# noinspection PyShadowingBuiltins
def builtin_floordiv(x, other):
    return x.__floordiv__(other)


# noinspection PyShadowingBuiltins
def builtin_rfloordiv(x, other):
    return x.__rfloordiv__(other)


# noinspection PyShadowingBuiltins
def builtin_abs(x):
    return x.__abs__()


# noinspection PyShadowingBuiltins
def builtin_float(x):
    return x.__float__()


# noinspection PyShadowingBuiltins
def builtin_int(x):
    return x.__int__()


# noinspection PyShadowingBuiltins
def builtin_bool(x):
    return x.__bool__()


# noinspection PyShadowingBuiltins
def builtin_lt(self: Union[ivy.Array, ivy.NativeArray],
                other: Union[int, float, ivy.Array, ivy.NativeArray]) \
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Computes the truth value of self_i < other_i for each element of an array instance with the respective element of the array other.

    Parameters
    self (array) : array instance. Should have a numeric data type.

    other (Union[int, float, array]) : other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

    Returns
    out (array) : an array containing the element-wise results. The returned array must have a data type of bool.
    """
    return self.__lt__(other)


# noinspection PyShadowingBuiltins
def builtin_le(x, other):
    return x.__le__(other)


# noinspection PyShadowingBuiltins
def builtin_eq(x, other):
    return x.__eq__(other)


# noinspection PyShadowingBuiltins
def builtin_ne(x, other):
    return x.__ne__(other)


# noinspection PyShadowingBuiltins
def builtin_gt(x, other):
    return x.__gt__(other)


# noinspection PyShadowingBuiltins
def builtin_ge(x, other):
    return x.__ge__(other)


# noinspection PyShadowingBuiltins
def builtin_and(x, other):
    return x.__and__(other)


# noinspection PyShadowingBuiltins
def builtin_rand(x, other):
    return x.__rand__(other)


# noinspection PyShadowingBuiltins
def builtin_or(x, other):
    return x.__or__(other)


# noinspection PyShadowingBuiltins
def builtin_ror(x, other):
    return x.__ror__(other)


# noinspection PyShadowingBuiltins
def builtin_invert(x, other):
    return x.__invert__(other)


# noinspection PyShadowingBuiltins
def builtin_xor(x, other):
    return x.__xor__(other)


# noinspection PyShadowingBuiltins
def builtin_rxor(x, other):
    return x.__rxor__(other)


# noinspection PyShadowingBuiltins
def builtin_deepcopy(x):
    return x.__deepcopy__()
