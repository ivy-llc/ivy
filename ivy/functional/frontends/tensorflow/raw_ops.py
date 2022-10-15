# global
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow import promote_types_of_tensorflow_inputs


def AddN(*, inputs, name="AddN"):
    inputs = ivy.array(inputs)
    return ivy.sum(inputs, axis=0, dtype=inputs.dtype)


def Acos(*, x, name="Acos"):
    return ivy.acos(x)


def Acosh(*, x, name="Acosh"):
    return ivy.acosh(x)


Add = tf_frontend.math.add


ArgMax = tf_frontend.math.argmax


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
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_and(x, y)


def BitwiseOr(*, x, y, name="BitwiseOr"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_or(x, y)


def BitwiseXor(*, x, y, name="BitwiseXor"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_xor(x, y)


def BroadcastTo(*, input, shape, name="BroadcastTo"):
    return ivy.broadcast_to(input, shape=shape)


def Cholesky(*, input, name="Cholesky"):
    return ivy.astype(ivy.cholesky(input), input.dtype)


def Ceil(*, x, name=None):
    return ivy.ceil(x)


def Concat(*, concat_dim, values, name="Concat"):
    return ivy.concat(values, axis=concat_dim)


def Cos(*, x, name="Cos"):
    return ivy.cos(x)


def Cosh(*, x, name="cosh"):
    return ivy.cosh(x)


Div = tf_frontend.math.divide


Cumprod = tf_frontend.math.cumprod


Cumsum = tf_frontend.math.cumsum


def Equal(*, x, y, incompatible_shape_error=True, name="Equal"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    if incompatible_shape_error:
        return ivy.equal(x, y)

    try:
        return ivy.equal(x, y)
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
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.floor_divide(x, y)


def Greater(*, x, y, name="Greater"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.greater(x, y)


def GreaterEqual(*, x, y, name="GreaterEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.greater_equal(x, y)


def Identity(*, input, name="Identity"):
    return ivy.copy_array(input)


def IdentityN(*, input, name="IdentityN"):
    return [ivy.copy_array(x) for x in input]


def Inv(*, x, name="Inv"):
    return ivy.astype(ivy.reciprocal(x), x.dtype)


def Invert(*, x, name="Invert"):
    return ivy.bitwise_invert(x)


def InvGrad(*, y, dy, name="InvGrad"):
    return ivy.multiply(ivy.negative(dy), ivy.multiply(y, y))


def LeftShift(*, x, y, name="LeftShift"):
    return ivy.bitwise_left_shift(x, y)


def Less(*, x, y, name="Less"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.less(x, y)


def LessEqual(*, x, y, name="LessEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.less_equal(x, y)


def Log(*, x, name="Log"):
    return ivy.log(x)


def LogicalOr(*, x, y, name="LogicalOr"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.logical_or(x, y)


def LogicalNot(*, x, name="LogicalNot"):
    return ivy.logical_not(x)


def MatMul(*, a, b, transpose_a=False, transpose_b=False, name="MatMul"):
    a, b = promote_types_of_tensorflow_inputs(a, b)
    return ivy.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


Maximum = tf_frontend.math.maximum


MatrixDeterminant = tf_frontend.linalg.det


def Max(*, input, axis, keep_dims=False, name="Max"):
    return ivy.astype(ivy.max(input, axis=axis, keepdims=keep_dims), input.dtype)


def Min(*, input, axis, keep_dims=False, name="Min"):
    return ivy.astype(ivy.min(input, axis=axis, keepdims=keep_dims), input.dtype)


def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


Neg = tf_frontend.math.negative


Mul = tf_frontend.math.multiply


def NotEqual(*, x, y, incompatible_shape_error=True, name="NotEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    if incompatible_shape_error:
        return ivy.not_equal(x, y)

    try:
        return ivy.not_equal(x, y)
    except (ivy.exceptions.IvyError, ivy.exceptions.IvyBackendException):
        return ivy.array(True)


def NthElement(*, input, n, reverse=False, name="NthElement"):
    return ivy.astype(ivy.sort(input, descending=reverse)[..., n], input.dtype)


def OnesLike(*, x, name="OnesLike"):
    return ivy.ones_like(x)


def Relu(features, name="Relu"):
    return ivy.relu(features)


def Reshape(*, tensor, shape, name="Reshape"):
    return ivy.reshape(tensor, shape)


def RightShift(*, x, y, name="RightShift"):
    return ivy.bitwise_right_shift(x, y)


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


Sub = tf_frontend.math.subtract


def Sum(*, input, axis, keep_dims=False, name="Sum"):
    return ivy.astype(ivy.sum(input, axis=axis, keepdims=keep_dims), input.dtype)


def Tan(*, x, name="Tan"):
    return tf_frontend.math.tan(x)


def Tanh(*, x, name="Tanh"):
    return ivy.tanh(x)


def Transpose(*, x, perm, name="Transpose"):
    ret = ivy.permute_dims(x, axes=perm)
    return ret


def TruncateDiv(*, x, y, name="TruncateDiv"):
    return ivy.astype(ivy.trunc_divide(x, y), x.dtype)


def ZerosLike(*, x, name="ZerosLike"):
    return ivy.zeros_like(x)


def Mean(*, input, axis, keep_dims=False, name="Mean"):
    return ivy.astype(ivy.mean(input, axis=axis, keepdims=keep_dims), input.dtype)
