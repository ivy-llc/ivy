# global
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow import check_tensorflow_casting
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    map_raw_ops_alias,
    to_ivy_dtype,
)

from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.utils.exceptions import IvyNotImplementedException

AddN = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add_n))


Acos = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.acos))


Acosh = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.acosh))


Add = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add))


ArgMax = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {"2.13.0 and below": ("complex",)},
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.argmax, kwargs_to_update={"dimension": "axis"}
        )
    )
)


AddV2 = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add))


Atan2 = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("float16")
        }, 
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.atan2)))


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "float16",
            "bool",
            "bfloat16",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def ApproximateEqual(
    *,
    x,
    y,
    tolerance=1e-05,
    name="ApproximateEqual",
):
    x, y = check_tensorflow_casting(x, y)
    return ivy.abs(x - y) < tolerance


@to_ivy_arrays_and_back
def Angle(
    *,
    input,
    Tout=ivy.float32,
    name="Angle",
):
    Tout = ivy.as_ivy_dtype(Tout) if Tout is not None else ivy.float32
    return ivy.astype(ivy.angle(input), Tout)


ArgMin = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {"2.13.0 and below": ("complex",)},
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.argmin, kwargs_to_update={"dimension": "axis"}
        )
    )
)


Asin = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.asin))


Atan = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.atan))


@to_ivy_arrays_and_back
def Atanh(*, x, name="Atanh"):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.bitwise_and(x, y)


@to_ivy_arrays_and_back
def BitwiseOr(*, x, y, name="BitwiseOr"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.bitwise_or(x, y)


@to_ivy_arrays_and_back
def BitwiseXor(*, x, y, name="BitwiseXor"):
    x, y = check_tensorflow_casting(x, y)
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


Cos = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cos))


Cosh = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cosh))


@to_ivy_arrays_and_back
def Cross(*, a, b, name="Cross"):
    a, b = check_tensorflow_casting(a, b)
    return ivy.cross(a, b)


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
    x, y = check_tensorflow_casting(x, y)
    if incompatible_shape_error:
        return ivy.equal(x, y)

    try:
        return ivy.equal(x, y)
    except (ivy.utils.exceptions.IvyError, ivy.utils.exceptions.IvyBackendException):
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
    x, y = check_tensorflow_casting(x, y)
    return ivy.floor_divide(x, y)


@to_ivy_arrays_and_back
def FloorMod(*, x, y, name="FloorMod"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.remainder(x, y)


@to_ivy_arrays_and_back
def FFT(*, input, name="FFT"):
    return ivy.astype(ivy.fft(input, -1), input.dtype)


@to_ivy_arrays_and_back
def Gather(*, params, indices, validate_indices=None, name="Gather"):
    return ivy.gather(params, indices, axis=0, batch_dims=0)


@to_ivy_arrays_and_back
def Greater(*, x, y, name="Greater"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.greater(x, y)


@to_ivy_arrays_and_back
def GreaterEqual(*, x, y, name="GreaterEqual"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.greater_equal(x, y)


Identity = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.identity)
)


IdentityN = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.identity_n)
)


@to_ivy_arrays_and_back
def Inv(*, x, name="Inv"):
    return ivy.astype(ivy.reciprocal(x), x.dtype)


Reciprocal = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.reciprocal))


@to_ivy_arrays_and_back
def Reverse(*, tensor, dims, name="Reverse"):
    ret = tensor
    for dim in enumerate(dims):
        if dim[1]:
            ret = ivy.flip(ret, axis=dim[0])
    return ret


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
    x, y = check_tensorflow_casting(x, y)
    return ivy.less(x, y)


LessEqual = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.less_equal))
)


@to_ivy_arrays_and_back
def Log(*, x, name="Log"):
    return ivy.log(x)


Log1p = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.log1p))

LogSoftmax = to_ivy_arrays_and_back(
    with_supported_dtypes(
        {
            "2.13.0 and below": (
                "bfloat16",
                "float32",
                "float64",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.log_softmax))
)

LogicalOr = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.logical_or))


@to_ivy_arrays_and_back
def LogicalNot(*, x, name="LogicalNot"):
    return ivy.logical_not(x)


@to_ivy_arrays_and_back
def MatMul(*, a, b, transpose_a=False, transpose_b=False, name="MatMul"):
    a, b = check_tensorflow_casting(a, b)
    return ivy.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


@to_ivy_arrays_and_back
def Rsqrt(*, x, name="Rsqrt"):
    return ivy.sqrt(ivy.reciprocal(x))


@to_ivy_arrays_and_back
def MatrixInverse(*, input, adjoint=False, name="MatrixInverse"):
    return ivy.inv(input, adjoint=adjoint)


MatrixDeterminant = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.linalg.det))


Max = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex",),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.reduce_max,
            kwargs_to_update={
                "input": "input_tensor",
                "keep_dims": "keepdims",
            },
        )
    )
)


Maximum = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.maximum))
)


Min = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex",),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.reduce_min,
            kwargs_to_update={
                "input": "input_tensor",
                "keep_dims": "keepdims",
            },
        )
    )
)


@to_ivy_arrays_and_back
def Minimum(*, x, y, name="Minimum"):
    return ivy.minimum(x, y)


Mul = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.multiply))


Neg = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.negative))


@to_ivy_arrays_and_back
def NotEqual(*, x, y, incompatible_shape_error=True, name="NotEqual"):
    x, y = check_tensorflow_casting(x, y)
    if incompatible_shape_error:
        return ivy.not_equal(x, y)

    try:
        return ivy.not_equal(x, y)
    except (ivy.utils.exceptions.IvyError, ivy.utils.exceptions.IvyBackendException):
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


@to_ivy_arrays_and_back
def Pad(*, input, paddings, name="Pad"):
    return ivy.constant_pad(input, paddings.to_list())


@to_ivy_arrays_and_back
def PadV2(*, input, paddings, constant_values, name="PadV2"):
    return ivy.constant_pad(input, paddings.to_list(), value=constant_values)


Relu = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex", "float16"),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.nn.relu))
)


RealDiv = to_ivy_arrays_and_back(
    with_supported_dtypes(
        {
            "2.13.0 and below": (
                "complex",
                "bfloat16",
                "float16",
                "float64",
                "float32",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.general_functions.realdiv))
)


Reshape = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.reshape)
)


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


ShapeN = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.shape_n)
)


Sin = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.sin))


@to_ivy_arrays_and_back
def Sinh(*, x, name="Sinh"):
    return ivy.sinh(x)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("unsigned",)},
    "tensorflow",
)
@to_ivy_arrays_and_back
def Sign(*, x, name="Sign"):
    return ivy.sign(x, np_variant=False)


Size = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.general_functions.size))


Split = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.split, kwargs_to_update={"num_split": "num_or_size_splits"}
    )
)


@to_ivy_arrays_and_back
def SplitV(*, value, size_splits, axis, num_split, name="SplitV"):
    return ivy.split(value, num_or_size_splits=size_splits, axis=axis)


@to_ivy_arrays_and_back
def Sqrt(*, x, name="Sqrt"):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def Square(*, x, name="Square"):
    return ivy.square(x)


Squeeze = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.squeeze)
)


Sub = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.subtract))


@to_ivy_arrays_and_back
def Sum(*, input, axis, keep_dims=False, name="Sum"):
    return ivy.astype(ivy.sum(input, axis=axis, keepdims=keep_dims), input.dtype)


Tan = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.tan))


Tanh = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.tanh))


@to_ivy_arrays_and_back
def TanhGrad(*, y, dy, name="TanhGrad"):
    return ivy.multiply(dy, ivy.subtract(1, ivy.multiply(y, y)))


@to_ivy_arrays_and_back
def Transpose(*, x, perm, name="Transpose"):
    ret = ivy.permute_dims(x, axes=perm)
    return ret


Cumsum = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cumsum))


@to_ivy_arrays_and_back
def TruncateDiv(*, x, y, name="TruncateDiv"):
    return ivy.astype(ivy.trunc_divide(x, y), x.dtype)


@with_unsupported_dtypes({"2.13.0 and below": ("float16", "bfloat16")}, "tensorflow")
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


Pow = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.pow))


Relu6 = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("complex", "float16"),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.nn.relu6,
        )
    )
)


Sigmoid = to_ivy_arrays_and_back(
    map_raw_ops_alias(tf_frontend.keras.activations.sigmoid)
)


Softmax = to_ivy_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.13.0 and below": ("float16",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.nn.softmax))
)


@to_ivy_arrays_and_back
def Softplus(*, features, name="Softplus"):
    return ivy.softplus(features)


@to_ivy_arrays_and_back
def Xdivy(*, x, y, name="Xdivy"):
    if (x == 0).all():
        return 0.0
    return ivy.divide(x, y)


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def Xlog1py(*, x, y, name="Xlog1py"):
    if (x == 0).all():
        return 0.0
    return ivy.multiply(x, ivy.log1p(y))


Xlogy = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.math.xlogy))


@to_ivy_arrays_and_back
def EuclideanNorm(*, input, axis, keep_dims=False, name="EuclideanNorm"):
    return ivy.astype(
        ivy.vector_norm(input, axis=axis, keepdims=keep_dims), input.dtype
    )


ConcatV2 = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.concat))


def _tf_to_ivy_ivy_arguments_for_conv(
    padding, ex_pading, strides, dilations, data_format
):
    if data_format.find("C") == 1:
        strides = strides[2:]
        dilations = dilations[2:]
        data_format = "channel_first"
        pad_index = [4, 8]
    else:
        strides = strides[1:-1]
        dilations = dilations[1:-1]
        data_format = "channel_last"
        pad_index = [2, 6]
    if padding == "EXPLICIT":
        padding = [
            (ex_pading[i], ex_pading[i + 1])
            for i in range(pad_index[0], pad_index[1], 2)
        ]
    return padding, strides, dilations, data_format


@to_ivy_arrays_and_back
def Conv2D(
    *,
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu,
    explicit_paddings,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name="Conv2D",
):
    padding, strides, dilations, data_format = _tf_to_ivy_ivy_arguments_for_conv(
        padding, explicit_paddings, strides, dilations, data_format
    )
    return ivy.conv_general_dilated(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        dims=2,
    )


@to_ivy_arrays_and_back
def Conv3D(
    *,
    input,
    filter,
    strides,
    padding,
    data_format="NDHWC",
    dilations=[1, 1, 1, 1, 1],
    name="Conv3D",
):
    # ivy.backends.tensorflow expects strides and dilations to be
    # a single integer value or a list of 3 values whereas the raw op
    # expects a list of 5 values
    if data_format == "NDHWC":
        strides = strides[1:-1]
        dilations = dilations[1:-1]
    elif data_format == "NCDHW":
        strides = strides[2:]
        dilations = dilations[2:]

    return tf_frontend.nn.conv3d(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )


@to_ivy_arrays_and_back
def Elu(features, name=None):
    zeros = ivy.zeros_like(features, dtype=ivy.dtype(features))
    ones = ivy.ones_like(features, dtype=ivy.dtype(features))
    ret_val = ivy.where(
        # if x > 0 => x; else e^x - 1
        features > zeros,
        features,
        ivy.subtract(ivy.exp(features), ones),
    )
    return ret_val


Elu.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "bfloat16",
        "float32",
        "float64",
    ),
    "jax": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
}


@to_ivy_arrays_and_back
def LinSpace(*, start, stop, num, name=None):
    return ivy.linspace(start, stop, num)


Roll = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.roll))


@to_ivy_arrays_and_back
def CumulativeLogsumexp(
    x, axis, exclusive=False, reverse=False, name="CumulativeLogsumexp"
):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def Complex(real, imag, Tout=ivy.complex64, name="Complex"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def AccumulateNV2(inputs, shape, name="AccumulateNV2"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def DebugGradientIdentity(input, name="DebugGradientIdentity"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def Real(input, Tout=ivy.float32, name="Real"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def BandedTriangularSolve(
    matrix,
    rhs,
    lower=True,
    adjoint=False,
    name="BandedTriangularSolve",
):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def BatchMatMul(x, y, adj_x=False, adj_y=False, name="BatchMatMul"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def BatchMatMulV2(x, y, adj_x=False, adj_y=False, name="BatchMatMulV2"):
    # TODO
    raise IvyNotImplementedException


@to_ivy_arrays_and_back
def BatchMatMulV3(x, y, Tout=ivy.Dtype, adj_x=False, adj_y=False, name="BatchMatMulV3"):
    # TODO
    raise IvyNotImplementedException


Slice = to_ivy_arrays_and_back(map_raw_ops_alias(tf_frontend.slice))

LeakyRelu = to_ivy_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.nn.leaky_relu,
    )
)

LeakyRelu.supported_dtypes = {
    "numpy": (
        "float32",
        "float64",
    ),
    "tensorflow": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "float32",
        "float64",
    ),
    "jax": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
}


@to_ivy_arrays_and_back
def Prod(*, input, axis, keep_dims=False, name="Prod"):
    return ivy.astype(ivy.prod(input, axis=axis, keepdims=keep_dims), input.dtype)


Zeta = to_ivy_arrays_and_back(
    with_supported_dtypes(
        {
            "2.13.0 and below": ("float32", "float64"),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.zeta))
)


@to_ivy_arrays_and_back
def Imag(
    *,
    input,
    Tout=ivy.float32,
    name="Imag",
):
    Tout = ivy.as_ivy_dtype(Tout) if Tout is not None else ivy.float32
    return ivy.astype(ivy.imag(input), Tout)


Imag.supported_dtypes = {
    "tensorflow": (
        "complex64",
        "complex128",
    ),
}
