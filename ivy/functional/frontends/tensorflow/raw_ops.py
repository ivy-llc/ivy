# global
import ivy


def AddN(*, inputs, name="AddN"):
    inputs = ivy.array(inputs)
    return ivy.sum(inputs, axis=0, dtype=inputs.dtype)


def Acos(*, x, name="Acos"):
    return ivy.acos(x)


def Acosh(*, x, name="Acosh"):
    return ivy.acosh(x)


def ArgMax(*, input, dimension, output_type=None, name=None):
    if output_type in ["uint16", "int16", "int32", "int64"]:
        return ivy.astype(ivy.argmax(input, axis=dimension), output_type)
    else:
        return ivy.astype(ivy.argmax(input, axis=dimension), "int64")


def ArgMin(*, input, dimension, output_type=None, name=None):
    if output_type in ["int32", "int64"]:
        return ivy.astype(ivy.argmin(input, axis=dimension), output_type)
    return ivy.astype(ivy.argmin(input, axis=dimension), "int64")


def Asin(*, x, name="asin"):
    return ivy.asin(x)


def Atan(*, x, name="atan"):
    return ivy.atan(x)


def Atanh(*, x, name="Atanh"):
    return ivy.atanh(x)


def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    return ivy.bitwise_and(x, y)


def BitwiseOr(*, x, y, name="BitwiseOr"):
    return ivy.bitwise_or(x, y)


def BitwiseXor(*, x, y, name="BitwiseXor"):
    return ivy.bitwise_xor(x, y)


def BroadcastTo(*, input, shape, name="BroadcastTo"):
    return ivy.broadcast_to(input, shape=shape)


def Concat(*, concat_dim, values, name="Concat"):
    return ivy.concat(values, axis=concat_dim)


def Cos(*, x, name="Cos"):
    return ivy.cos(x)


def Cosh(*, x, name="cosh"):
    return ivy.cosh(x)


def Equal(*, x, y, incompatible_shape_error=True, name="Equal"):
    if incompatible_shape_error:
        return ivy.equal(x, y)

    try:
        ivy.equal(x, y)
    except (ivy.exceptions.IvyError, ivy.exceptions.IvyBackendException):
        return ivy.array(False)


def Exp(*, x, name="Exp"):
    return ivy.exp(x)


def Expm1(*, x, name="Expm1"):
    return ivy.expm1(x)


def Fill(*, dims, value, name="Full"):
    return ivy.full(dims, value)


def Floor(*, x, name="Floor"):
    return ivy.floor(x)


def FloorDiv(*, x, y, name="FloorDiv"):
    return ivy.floor_divide(x, y)


def Less(*, x, y, name="Less"):
    return ivy.less(x, y)


def LessEqual(*, x, y, name="LessEqual"):
    return ivy.less_equal(x, y)


def Log(*, x, name="Log"):
    return ivy.log(x)


def LogicalOr(*, x, y, name="LogicalOr"):
    return ivy.logical_or(x, y)


def LogicalNot(*, x, name="LogicalNot"):
    return ivy.logical_not(x)


def Maximum(*, x, y, name="Maximum"):
    return ivy.maximum(x, y)


def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


def Neg(*, x, name="Neg"):
    return ivy.negative(x)


def NotEqual(*, x, y, incompatible_shape_error=True, name="NotEqual"):
    if incompatible_shape_error:
        return ivy.not_equal(x, y)

    try:
        ivy.not_equal(x, y)
    except (ivy.exceptions.IvyError, ivy.exceptions.IvyBackendException):
        return ivy.array(False)


def Reshape(*, tensor, shape, name="Reshape"):
    return ivy.reshape(tensor, shape)


def Shape(*, input, output_type=ivy.int32, name="Shape"):
    return ivy.astype(ivy.shape(input, as_array=True), output_type, copy=False)


def Sin(*, x, name="Sin"):
    return ivy.sin(x)


def Sinh(*, x, name="Sinh"):
    return ivy.sinh(x)


def Sqrt(*, x, name="Sqrt"):
    return ivy.sqrt(x)


def Square(*, x, name="Square"):
    return ivy.square(x)


def Sub(*, x, y, name="Sub"):
    return ivy.subtract(x, y)


def Tan(*, x, name="Tan"):
    return ivy.tan(x)


def Tanh(*, x, name="Tanh"):
    return ivy.tanh(x)


def Transpose(*, x, perm, name="Transpose"):
    ret = ivy.permute_dims(x, axes=perm)
    return ret


def ZerosLike(*, x, name="ZerosLike"):
    return ivy.zeros_like(x)
