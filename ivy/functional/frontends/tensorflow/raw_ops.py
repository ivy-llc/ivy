# global
import ivy


def acos(x, name="acos"):
    return ivy.acos(x)


def acosh(x, name="acosh"):
    return ivy.acosh(x)


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis=axis)


def cos(x, name="cos"):
    return ivy.cos(x)


def cosh(x, name="cosh"):
    return ivy.cosh(x)


def fill(dims, value, name="full"):
    return ivy.full(dims, value)


def asin(x, name="asin"):
    return ivy.asin(x)


def atan(x, name="atan"):
    return ivy.atan(x)


def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    return ivy.bitwise_and(x, y)


def BitwiseXor(*, x, y, name="BitwiseXor"):
    return ivy.bitwise_xor(x, y)


def atanh(x, name="atanh"):
    return ivy.atanh(x)


def tan(x, name="tan"):
    return ivy.tan(x)


def tanh(x, name="tanh"):
    return ivy.tanh(x)


def sin(x, name="sin"):
    return ivy.sin(x)


def square(x, name="square"):
    return ivy.square(x)


def sqrt(x, name="sqrt"):
    return ivy.sqrt(x)


def Maximum(*, x, y, name="Maximum"):
    return ivy.maximum(x, y)


def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)
