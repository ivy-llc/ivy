# global
import ivy


def Acos(*, x, name="Acos"):
    return ivy.acos(x)


def Acosh(*, x, name="Acosh"):
    return ivy.acosh(x)


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis=axis)


def ArgMax(*, input, dimension, output_type=None, name=None):
    if output_type in ["uint16", "int16", "int32", "int64"]:
        return ivy.astype(ivy.argmax(input, axis=dimension), output_type)
    else:
        return ivy.astype(ivy.argmax(input, axis=dimension), "int64")


def Cos(*, x, name="Cos"):
    return ivy.cos(x)


def Cosh(*, x, name="cosh"):
    return ivy.cosh(x)


def fill(dims, value, name="full"):
    return ivy.full(dims, value)


def Asin(*, x, name="asin"):
    return ivy.asin(x)


def Atan(*, x, name="atan"):
    return ivy.atan(x)


def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    return ivy.bitwise_and(x, y)


def BitwiseOr(*, x, y, name="BitwiseOr"):
    return ivy.bitwise_or(x, y)


def BitwiseXor(*, x, y, name="BitwiseXor"):
    return ivy.bitwise_xor(x, y)


def Atanh(*, x, name="Atanh"):
    return ivy.atanh(x)


def Tan(*, x, name="Tan"):
    return ivy.tan(x)


def Tanh(*, x, name="Tanh"):
    return ivy.tanh(x)


def Sin(*, x, name="Sin"):
    return ivy.sin(x)


def Square(*, x, name="Square"):
    return ivy.square(x)


def Sqrt(*, x, name="Sqrt"):
    return ivy.sqrt(x)


def Maximum(*, x, y, name="Maximum"):
    return ivy.maximum(x, y)


def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


def Sub(*, x, y, name="Sub"):
    return ivy.subtract(x, y)


def Less(*, x, y, name="Less"):
    return ivy.less(x, y)


def LessEqual(*, x, y, name="LessEqual"):
    return ivy.less_equal(x, y)


def Floor(*, x, name="Floor"):
    return ivy.floor(x)


def FloorDiv(*, x, y, name="FloorDiv"):
    return ivy.floor_divide(x, y)


def Exp(*, x, name="Exp"):
    return ivy.exp(x)


def Expm1(*, x, name="Expm1"):
    return ivy.expm1(x)


def Log(*, x, name="Log"):
    return ivy.log(x)


def Reshape(tensor, shape, name="reshape"):
    return ivy.reshape(tensor, shape)


def Sinh(*, x, name="Sinh"):
    return ivy.sinh(x)
