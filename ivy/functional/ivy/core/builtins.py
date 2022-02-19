"""
Collection of builtin Ivy functions.
"""


# noinspection PyShadowingBuiltins
def dir(x):
    return x.__dir__()


# noinspection PyShadowingBuiltins
def getattr(x, item):
    return x.__getattr__(item)


# noinspection PyShadowingBuiltins
def getattribute(x, item):
    return x.__getattribute__(item)


# noinspection PyShadowingBuiltins
def getitem(x, query):
    return x.__getitem__(query)


# noinspection PyShadowingBuiltins
def setitem(x, query, val):
    return x.__setitem__(query, val)


# noinspection PyShadowingBuiltins
def contains(x, key):
    return x.__contains__(key)


# noinspection PyShadowingBuiltins
def pos(x):
    return x.__pos__()


# noinspection PyShadowingBuiltins
def neg(x):
    return x.__neg__()


# noinspection PyShadowingBuiltins
def pow(x, power):
    return x.__pow__(power)


# noinspection PyShadowingBuiltins
def rpow(x, power):
    return x.__rpow__(power)


# noinspection PyShadowingBuiltins
def add(x, other):
    return x.__add__(other)


# noinspection PyShadowingBuiltins
def radd(x, other):
    return x.__radd__(other)


# noinspection PyShadowingBuiltins
def sub(x, other):
    return x.__sub__(other)


# noinspection PyShadowingBuiltins
def rsub(x, other):
    return x.__rsub__(other)


# noinspection PyShadowingBuiltins
def mul(x, other):
    return x.__mul__(other)


# noinspection PyShadowingBuiltins
def rmul(x, other):
    return x.__rmul__(other)


# noinspection PyShadowingBuiltins
def truediv(x, other):
    return x.__truediv__(other)


# noinspection PyShadowingBuiltins
def rtruediv(x, other):
    return x.__rtruediv__(other)


# noinspection PyShadowingBuiltins
def floordiv(x, other):
    return x.__floordiv__(other)


# noinspection PyShadowingBuiltins
def rfloordiv(x, other):
    return x.__rfloordiv__(other)


# noinspection PyShadowingBuiltins
def abs(x):
    return x.__abs__()


# noinspection PyShadowingBuiltins
def float(x):
    return x.__float__()


# noinspection PyShadowingBuiltins
def int(x):
    return x.__int__()


# noinspection PyShadowingBuiltins
def bool(x):
    return x.__bool__()


# noinspection PyShadowingBuiltins
def lt(x, other):
    return x.__lt__(other)


# noinspection PyShadowingBuiltins
def le(x, other):
    return x.__le__(other)


# noinspection PyShadowingBuiltins
def eq(x, other):
    return x.__eq__(other)


# noinspection PyShadowingBuiltins
def ne(x, other):
    return x.__ne__(other)


# noinspection PyShadowingBuiltins
def gt(x, other):
    return x.__gt__(other)


# noinspection PyShadowingBuiltins
def ge(x, other):
    return x.__ge__(other)


# noinspection PyShadowingBuiltins
def and_(x, other):
    return x.__and__(other)


# noinspection PyShadowingBuiltins
def rand(x, other):
    return x.__rand__(other)


# noinspection PyShadowingBuiltins
def or_(x, other):
    return x.__or__(other)


# noinspection PyShadowingBuiltins
def ror(x, other):
    return x.__ror__(other)


# noinspection PyShadowingBuiltins
def invert(x, other):
    return x.__invert__(other)


# noinspection PyShadowingBuiltins
def xor(x, other):
    return x.__xor__(other)


# noinspection PyShadowingBuiltins
def rxor(x, other):
    return x.__rxor__(other)


# noinspection PyShadowingBuiltins
def deepcopy(x):
    return x.__deepcopy__()
