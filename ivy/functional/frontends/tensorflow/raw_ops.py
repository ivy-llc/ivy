# global
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend

from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.tensorflow import promote_types_of_tensorflow_inputs

from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def AddN(*, inputs, name="AddN"):
    inputs = ivy.array(inputs)
    return ivy.sum(inputs, axis=0, dtype=inputs.dtype)


@to_ivy_arrays_and_back
def Acos(*, x, name="Acos"):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def Acosh(*, x, name="Acosh"):
    return ivy.acosh(x)


Add = tf_frontend.math.add


ArgMax = tf_frontend.math.argmax


@to_ivy_arrays_and_back
def ArgMin(*, input, dimension, output_type=None, name=None):
    if output_type in ["int32", "int64"]:
        return ivy.astype(ivy.argmin(input, axis=dimension), output_type)
    return ivy.astype(ivy.argmin(input, axis=dimension), "int64")


@to_ivy_arrays_and_back
def Asin(*, x, name="asin"):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def Atan(*, x, name="atan"):
    return ivy.atan(x)


@to_ivy_arrays_and_back
def Atanh(*, x, name="Atanh"):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_and(x, y)


@to_ivy_arrays_and_back
def BitwiseOr(*, x, y, name="BitwiseOr"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_or(x, y)


@to_ivy_arrays_and_back
def BitwiseXor(*, x, y, name="BitwiseXor"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.bitwise_xor(x, y)


@to_ivy_arrays_and_back
def BroadcastTo(*, input, shape, name="BroadcastTo"):
    return ivy.broadcast_to(input, shape=shape)


@to_ivy_arrays_and_back
def Cholesky(*, input, name="Cholesky"):
    return ivy.astype(ivy.cholesky(input), input.dtype)


@to_ivy_arrays_and_back
def Ceil(*, x, name=None):
    return ivy.ceil(x)


def Concat(*, concat_dim, values, name="Concat"):
    return ivy.concat(values, axis=concat_dim)


@to_ivy_arrays_and_back
def Cos(*, x, name="Cos"):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def Cosh(*, x, name="cosh"):
    return ivy.cosh(x)


Div = tf_frontend.math.divide


def Diag(*, diagonal, name="Diag"):
    return ivy.astype(ivy.diag(diagonal), diagonal.dtype)


Cumprod = tf_frontend.math.cumprod


@to_ivy_arrays_and_back
def Equal(*, x, y, incompatible_shape_error=True, name="Equal"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    if incompatible_shape_error:
        return ivy.equal(x, y)

    try:
        return ivy.equal(x, y)
    except (ivy.exceptions.IvyError, ivy.exceptions.IvyBackendException):
        return ivy.array(False)


@to_ivy_arrays_and_back
def Exp(*, x, name="Exp"):
    return ivy.exp(x)


@to_ivy_arrays_and_back
def Expm1(*, x, name="Expm1"):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def Fill(*, dims, value, name="Full"):
    return ivy.full(dims, value)


@to_ivy_arrays_and_back
def Floor(*, x, name="Floor"):
    return ivy.floor(x)


@to_ivy_arrays_and_back
def FloorDiv(*, x, y, name="FloorDiv"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.floor_divide(x, y)


@to_ivy_arrays_and_back
def Gather(*, params, indices, validate_indices=None, name="Gather"):
    return ivy.gather(params, indices, axis=0, batch_dims=0)


@to_ivy_arrays_and_back
def Greater(*, x, y, name="Greater"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.greater(x, y)


@to_ivy_arrays_and_back
def GreaterEqual(*, x, y, name="GreaterEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.greater_equal(x, y)


@to_ivy_arrays_and_back
def Identity(*, input, name="Identity"):
    return ivy.copy_array(input)


@to_ivy_arrays_and_back
def IdentityN(*, input, name="IdentityN"):
    return [ivy.copy_array(x) for x in input]


@to_ivy_arrays_and_back
def Inv(*, x, name="Inv"):
    return ivy.astype(ivy.reciprocal(x), x.dtype)


@to_ivy_arrays_and_back
def Invert(*, x, name="Invert"):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def InvGrad(*, y, dy, name="InvGrad"):
    return ivy.multiply(ivy.negative(dy), ivy.multiply(y, y))


@to_ivy_arrays_and_back
def LeftShift(*, x, y, name="LeftShift"):
    return ivy.bitwise_left_shift(x, y)


@to_ivy_arrays_and_back
def Less(*, x, y, name="Less"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.less(x, y)


@to_ivy_arrays_and_back
def LessEqual(*, x, y, name="LessEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.less_equal(x, y)


@to_ivy_arrays_and_back
def Log(*, x, name="Log"):
    return ivy.log(x)


@to_ivy_arrays_and_back
def LogicalOr(*, x, y, name="LogicalOr"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.logical_or(x, y)


@to_ivy_arrays_and_back
def LogicalNot(*, x, name="LogicalNot"):
    return ivy.logical_not(x)


@to_ivy_arrays_and_back
def MatMul(*, a, b, transpose_a=False, transpose_b=False, name="MatMul"):
    a, b = promote_types_of_tensorflow_inputs(a, b)
    return ivy.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


def MatrixInverse(*, input, adjoint=False, name="MatrixInverse"):
    return ivy.inv(input, adjoint=adjoint)


MatrixDeterminant = tf_frontend.linalg.det


@to_ivy_arrays_and_back
def Max(*, input, axis, keep_dims=False, name="Max"):
    return ivy.astype(ivy.max(input, axis=axis, keepdims=keep_dims), input.dtype)


Maximum = tf_frontend.math.maximum


@to_ivy_arrays_and_back
def Min(*, input, axis, keep_dims=False, name="Min"):
    return ivy.astype(ivy.min(input, axis=axis, keepdims=keep_dims), input.dtype)


@to_ivy_arrays_and_back
def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


Mul = tf_frontend.math.multiply


Neg = tf_frontend.math.negative


@to_ivy_arrays_and_back
def NotEqual(*, x, y, incompatible_shape_error=True, name="NotEqual"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    if incompatible_shape_error:
        return ivy.not_equal(x, y)

    try:
        return ivy.not_equal(x, y)
    except (ivy.exceptions.IvyError, ivy.exceptions.IvyBackendException):
        return ivy.array(True)


@to_ivy_arrays_and_back
def NthElement(*, input, n, reverse=False, name="NthElement"):
    return ivy.astype(ivy.sort(input, descending=reverse)[..., n], input.dtype)


@to_ivy_arrays_and_back
def OnesLike(*, x, name="OnesLike"):
    return ivy.ones_like(x)


@to_ivy_arrays_and_back
def Relu(features, name="Relu"):
    return ivy.relu(features)


@to_ivy_arrays_and_back
def Reshape(*, tensor, shape, name="Reshape"):
    return ivy.reshape(tensor, shape)


@to_ivy_arrays_and_back
def RightShift(*, x, y, name="RightShift"):
    return ivy.bitwise_right_shift(x, y)


@to_ivy_arrays_and_back
def Round(*, x, name="Round"):
    return ivy.round(x)


@to_ivy_arrays_and_back
def Shape(*, input, output_type=ivy.int32, name="Shape"):
    return ivy.astype(ivy.shape(input, as_array=True), output_type, copy=False)


@to_ivy_arrays_and_back
def Sin(*, x, name="Sin"):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def Sinh(*, x, name="Sinh"):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def Sqrt(*, x, name="Sqrt"):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def Square(*, x, name="Square"):
    return ivy.square(x)


Sub = tf_frontend.math.subtract


@to_ivy_arrays_and_back
def Sum(*, input, axis, keep_dims=False, name="Sum"):
    return ivy.astype(ivy.sum(input, axis=axis, keepdims=keep_dims), input.dtype)


@to_ivy_arrays_and_back
def Tan(*, x, name="Tan"):
    return tf_frontend.math.tan(x)


@to_ivy_arrays_and_back
def Tanh(*, x, name="Tanh"):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def Transpose(*, x, perm, name="Transpose"):
    ret = ivy.permute_dims(x, axes=perm)
    return ret


Cumsum = tf_frontend.math.cumsum


@to_ivy_arrays_and_back
def TruncateDiv(*, x, y, name="TruncateDiv"):
    return ivy.astype(ivy.trunc_divide(x, y), x.dtype)


@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
@to_ivy_arrays_and_back
def Unpack(*, value, num, axis=0, name="Unpack"):
    return ivy.unstack(value, axis=axis)[:num]


@to_ivy_arrays_and_back
def ZerosLike(*, x, name="ZerosLike"):
    return ivy.zeros_like(x)


@to_ivy_arrays_and_back
def Mean(*, input, axis, keep_dims=False, name="Mean"):
    return ivy.astype(ivy.mean(input, axis=axis, keepdims=keep_dims), input.dtype)


@to_ivy_arrays_and_back
def Pow(*, x, y, name="Pow"):
    return ivy.pow(x, y)


def Relu6(features, name="Relu6"):
    return ivy.clip(features, 0, 6)


@to_ivy_arrays_and_back
def Sigmoid(x, name="Sigmoid"):
    return ivy.sigmoid(x)
