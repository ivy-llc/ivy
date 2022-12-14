# global
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend

from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    map_raw_ops_alias,
    to_ivy_dtype,
)
from ivy.functional.frontends.tensorflow import (
    promote_types_of_tensorflow_inputs,
)

from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def AddN(*, inputs, name="AddN"):
    return ivy.sum(inputs, dtype=inputs.dtype)


@to_ivy_arrays_and_back
def Acos(*, x, name="Acos"):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def Acosh(*, x, name="Acosh"):
    return ivy.acosh(x)


Add = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add))


ArgMax = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.argmax,
        kwargs_to_update={"dimension": "axis"},
    )
)


@to_ivy_arrays_and_back
def ArgMin(*, input, dimension, output_type=None, name=None):
    output_type = to_ivy_dtype(output_type)
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


@to_ivy_arrays_and_back
def Concat(*, concat_dim, values, name="Concat"):
    return ivy.concat(values, axis=concat_dim)


@to_ivy_arrays_and_back
def Cos(*, x, name="Cos"):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def Cosh(*, x, name="Cosh"):
    return ivy.cosh(x)


Div = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.divide))


@to_ivy_arrays_and_back
def Diag(*, diagonal, name="Diag"):
    return ivy.astype(ivy.diag(diagonal), diagonal.dtype)


Cumprod = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cumprod))


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
def Reciprocal(*, x, name=None):
    return ivy.reciprocal(x)


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


@to_ivy_arrays_and_back
def MatrixInverse(*, input, adjoint=False, name="MatrixInverse"):
    return ivy.inv(input, adjoint=adjoint)


MatrixDeterminant = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.linalg.det))


Max = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.reduce_max,
        kwargs_to_update={
            "input": "input_tensor",
            "keep_dims": "keepdims",
        },
    )
)


Maximum = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.maximum,
        kwargs_to_update={"x": "a", "y": "b"},
    )
)


Min = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.reduce_min,
        kwargs_to_update={
            "input": "input_tensor",
            "keep_dims": "keepdims",
        },
    )
)


@to_ivy_arrays_and_back
def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


Mul = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.multiply))


Neg = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.negative))


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
def Pack(*, values, axis=0, name="Pack"):
    return ivy.stack(values, axis=axis)


Relu = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.keras.activations.relu,
        kwargs_to_update={"features": "x"},
    )
)


@to_ivy_arrays_and_back
def RealDiv(*, x, y, name="RealDiv"):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.divide(x, y)


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
    output_type = to_ivy_dtype(output_type)
    return ivy.astype(ivy.shape(input, as_array=True), output_type, copy=False)


@to_ivy_arrays_and_back
def Sin(*, x, name="Sin"):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def Sinh(*, x, name="Sinh"):
    return ivy.sinh(x)


@with_unsupported_dtypes(
    {
        "2.10.0 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def Sign(*, x, name="Sign"):
    return ivy.sign(x)


@to_ivy_arrays_and_back
def Split(*, axis, value, num_split, name="Split"):
    return ivy.split(value, num_or_size_splits=num_split, axis=axis)


@to_ivy_arrays_and_back
def SplitV(*, value, size_splits, axis, num_split, name="SplitV"):
    return ivy.split(value, num_or_size_splits=size_splits, axis=axis)


@to_ivy_arrays_and_back
def Sqrt(*, x, name="Sqrt"):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def Square(*, x, name="Square"):
    return ivy.square(x)


@to_ivy_arrays_and_back
def Squeeze(*, input, axis, name="Squeeze"):
    return ivy.squeeze(input, axis=axis)


Sub = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.subtract))


@to_ivy_arrays_and_back
def Sum(*, input, axis, keep_dims=False, name="Sum"):
    return ivy.astype(ivy.sum(input, axis=axis, keepdims=keep_dims), input.dtype)


Tan = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.tan))


@to_ivy_arrays_and_back
def Tanh(*, x, name="Tanh"):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def Transpose(*, x, perm, name="Transpose"):
    ret = ivy.permute_dims(x, axes=perm)
    return ret


Cumsum = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cumsum))


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


Mean = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.reduce_mean,
        kwargs_to_update={
            "input": "input_tensor",
            "keep_dims": "keepdims",
        },
    )
)


@to_ivy_arrays_and_back
def Pow(*, x, y, name="Pow"):
    return ivy.pow(x, y)


@to_ivy_arrays_and_back
def Relu6(features, name="Relu6"):
    return ivy.clip(features, 0, 6)


Sigmoid = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.keras.activations.sigmoid)
)


@to_ivy_arrays_and_back
def Softplus(features, name="Softplus"):
    return ivy.softplus(features)


@to_ivy_arrays_and_back
def Xdivy(*, x, y, name="Xdivy"):
    if (x == 0).all():
        return 0.0
    return ivy.divide(x, y)


@with_unsupported_dtypes({"2.10.0 and below": ("bfloat16")}, "tensorflow")
@to_ivy_arrays_and_back
def Xlog1py(*, x, y, name="Xlog1py"):
    if (x == 0).all():
        return 0.0
    return ivy.multiply(x, ivy.log1p(y))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.10.0 and below": ("bfloat16")}, "tensorflow")
def Xlogy(*, x, y, name="Xlogy"):
    if (x == 0).all():
        return 0.0
    return ivy.multiply(x, ivy.log(y))
