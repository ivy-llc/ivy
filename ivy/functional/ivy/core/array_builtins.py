"""
Collection of builtin Ivy functions.
"""


# noinspection PyShadowingBuiltins
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
def builtin_pow(x, power):
    return x.__pow__(power)


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
def builtin_lt(x, other):
    return x.__lt__(other)


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
