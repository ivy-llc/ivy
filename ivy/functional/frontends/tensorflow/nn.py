# global
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.tensorflow import math
import ivy.functional.frontends.tensorflow as tf_frontend


@to_ivy_arrays_and_back
def atrous_conv2d(value, filters, rate, padding):
    return ivy.conv2d(value, filters, 1, padding, dilations=rate)


@to_ivy_arrays_and_back
def atrous_conv2d_transpose(value, filters, output_shape, rate, padding):
    return ivy.conv2d_transpose(
        value, filters, rate, padding, output_shape=output_shape, dilations=rate
    )


@to_ivy_arrays_and_back
def conv1d(
    input, filters, stride, padding, data_format="NWC", dilations=None, name=None
):
    return ivy.conv1d(
        input, filters, stride, padding, data_format=data_format, dilations=dilations
    )


@to_ivy_arrays_and_back
def conv1d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NWC",
    dilations=None,
    name=None,
):
    return ivy.conv1d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
def gelu(features, approximate=False, name=None):
    return ivy.gelu(features, approximate=approximate)


@to_ivy_arrays_and_back
def conv2d(
    input, filters, strides, padding, data_format="NHWC", dilations=None, name=None
):
    return ivy.conv2d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


@to_ivy_arrays_and_back
def conv2d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NHWC",
    dilations=None,
    name=None,
):
    return ivy.conv2d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
def conv3d(
    input, filters, strides, padding, data_format="NDHWC", dilations=None, name=None
):
    return ivy.conv3d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


@to_ivy_arrays_and_back
def conv3d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NDHWC",
    dilations=None,
    name=None,
):
    return ivy.conv3d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
    inv = 1.0 / ivy.sqrt(variance + variance_epsilon)
    if scale is not None:
        inv *= scale

    return x * ivy.astype(inv, x.dtype, copy=False) + ivy.astype(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype
    )


@to_ivy_arrays_and_back
def dropout(x, prob, scale, dtype, name=None):
    return ivy.dropout(x, prob, scale, dtype)


@to_ivy_arrays_and_back
def silu(features, beta: float = 1.0):
    beta = ivy.astype(ivy.array(beta), ivy.dtype(features))
    return ivy.multiply(features, ivy.sigmoid(ivy.multiply(beta, features)))


silu.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
    "bfloat16",
)


@to_ivy_arrays_and_back
def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    ivy.assertions.check_shape(labels, logits)
    zeros = ivy.zeros_like(logits)
    max_logits = ivy.where(logits >= zeros, logits, zeros)
    neg_abs_logits = ivy.negative(ivy.abs(logits))
    neg_multiple = ivy.negative(ivy.multiply(logits, labels))
    ret_val = ivy.add(max_logits, neg_multiple)
    return ivy.add(ret_val, ivy.log1p(ivy.exp(neg_abs_logits)))


sigmoid_cross_entropy_with_logits.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
)


@to_ivy_arrays_and_back
def weighted_cross_entropy_with_logits(
    labels=None, logits=None, pos_weight=1.0, name=None
):
    ivy.assertions.check_shape(labels, logits)
    ones = ivy.ones_like(labels)
    zeros = ivy.zeros_like(logits)
    log_weight = ivy.add(ones, ivy.multiply(pos_weight - 1, labels))
    ones_minus_labels = ivy.subtract(ones, labels)
    first_term = ivy.multiply(ones_minus_labels, logits)

    max_neg_logits = ivy.where(
        ivy.negative(logits) >= zeros, ivy.negative(logits), zeros
    )
    neg_abs_logits = ivy.negative(ivy.abs(logits))
    log_neg_abs_logits = ivy.log1p(ivy.exp(neg_abs_logits))
    second_term = ivy.multiply(log_weight, ivy.add(log_neg_abs_logits, max_neg_logits))
    return ivy.add(first_term, second_term)


weighted_cross_entropy_with_logits.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
)


@with_supported_dtypes(
    {"2.9.0 and below": ("float32", "float16", "bfloat16")}, "tensorflow"
)
@to_ivy_arrays_and_back
def local_response_normalization(
    input, /, *, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None
):
    input_shape = ivy.shape(input)
    ivy.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message="4D input, but got input with sizes " + str(input_shape),
    )
    input_perm = ivy.astype(ivy.permute_dims(input, axes=[0, 3, 1, 2]), input.dtype)
    bias = ivy.astype(ivy.array(bias), input.dtype)
    alpha = ivy.astype(ivy.array(alpha), input.dtype)
    beta = ivy.astype(ivy.array(beta), input.dtype)
    sqr_sum = ivy.astype(ivy.zeros_like(input_perm), input.dtype)
    for p in range(input_shape[0]):
        sqr_sum[p] = [
            sum(
                ivy.pow(
                    input_perm[p][max(c - depth_radius, 0) : c + depth_radius + 1], 2.0
                )
            )
            for c in range(input_shape[3])
        ]
    div = ivy.multiply(input_perm, ivy.pow(math.add(sqr_sum * alpha, bias), -beta))
    return ivy.permute_dims(div, [0, 2, 3, 1])


@to_ivy_arrays_and_back
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    return ivy.max_pool1d(input, ksize, strides, padding, data_format=data_format)


@to_ivy_arrays_and_back
def moments(x, axes, shift=None, keepdims=False, name=None):
    return ivy.mean(x, axis=axes, keepdims=keepdims), ivy.var(
        x, axis=axes, keepdims=keepdims
    )


@to_ivy_arrays_and_back
def bias_add(value, bias, data_format=None, name=None):
    if data_format is None:
        data_format = "N...C"

    if data_format == "N...C":
        return ivy.add(value, bias)
    else:
        value = ivy.swapaxes(value, 1, -1)
        res = ivy.add(value, bias)
        return ivy.swapaxes(res, 1, -1)


def _convolution_broadcast_helper(
    arg, num_spatial_dims, channel_index, name="dilations"
):
    # Helper to broadcast dilations and strides to correct dims
    if arg is None:
        return [1] * num_spatial_dims
    else:
        # Broadcast to rcorrect dimensions
        arg = list(arg)
        len_arg = len(arg)
        if len_arg == 1:
            return arg * num_spatial_dims
        elif len_arg == num_spatial_dims:
            # Add dimensions for batch and channel
            if channel_index == 1:
                return [1, 1] + arg
            else:
                return [1] + arg + [1]
        elif len_arg != num_spatial_dims + 2:
            raise ValueError(
                f"{name} should be of length 1, "
                f"{num_spatial_dims} or {num_spatial_dims + 2}. "
                f"Received: {name}={arg} of length {len_arg}."
            )


@to_ivy_arrays_and_back
def convolution(
    input,
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
):

    DATA_FORMATS = frozenset(
        {
            "NWC",
            "NHC",
            "NHWC",
            "NWHC",
            "NDHWC",
            "NDWHC",
            "NHDWC",
            "NHWDC",
            "NWDHC",
            "NWHDC",
        }
    )
    ALLOWED_NUM_SPATIAL_DIMS = [1, 2, 3]
    PADDINGS = ["VALID", "SAME"]

    # Perform necessary assertions first
    input_depth = ivy.shape(input)[-1]
    filters_depth = ivy.shape(filters)[-2]

    # Inconsistent input and filter depths
    if input_depth != filters_depth:
        raise ValueError(
            f"`input` and `filter` must have the same depth: "
            f"{input_depth} vs {filters_depth}."
        )

    # Figure out N
    input_rank = tf_frontend.rank(input).numpy()
    num_spatial_dims = input_rank - 2

    # Incompatible N-D convolution
    if num_spatial_dims not in ALLOWED_NUM_SPATIAL_DIMS:
        raise ValueError(
            "`num_spatial_dims` must be 1, 2, or 3. "
            f"Received: num_spatial_dims={num_spatial_dims}."
        )

    # Incompatible padding
    if padding not in PADDINGS:
        raise ValueError(
            f"Value for attr `padding` is not in the list of allowed values: {PADDINGS}"
        )

    # The number of dimensions corresponding to num_batches
    num_batch_dims = input_rank - num_spatial_dims - 1

    # Figure out the channel_index
    if data_format is None or data_format in DATA_FORMATS:
        channel_index = num_batch_dims + num_spatial_dims
    else:
        channel_index = num_batch_dims

    # Dilations
    dilations = _convolution_broadcast_helper(
        dilations, num_spatial_dims, channel_index, name="dilations"
    )

    # Strides
    strides = _convolution_broadcast_helper(
        strides, num_spatial_dims, channel_index, name="strides"
    )

    if channel_index == 1:
        strides = strides[2:]
        dilations = dilations[2:]
    else:
        strides = strides[1:-1]
        dilations = dilations[1:-1]

    # Flag for atrous convolution
    is_atrous_conv = any(i != 1 for i in dilations)

    # Output shape
    if padding == "SAME":
        output_shape = ivy.ceil(list(input.shape[2:]) / strides)
    elif padding == "VALID":
        output_shape = ivy.ceil(
            (list(input.shape[2:]) - (list(filter.shape[:-2]) - 1) * dilations)
            / strides
        )

    # Transpose convolutions
    if data_format.startswith("NC"):
        input = ivy.permute_dims(
            input, axes=[0] + [*range(2, num_spatial_dims + 2)] + [1]
        )

        # conv1d_transpose
        if num_spatial_dims == 1:
            output = tf_frontend.nn.conv1d_transpose(
                input,
                filters,
                output_shape,
                strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
        # atrous_conv2d_transpose
        elif num_spatial_dims == 2 and is_atrous_conv:
            output = tf_frontend.nn.atrous_conv2d_transpose(
                input, filters, output_shape, dilations, padding
            )
        # conv2d_transpose
        elif num_spatial_dims == 2 and not is_atrous_conv:
            output = tf_frontend.nn.conv2d_transpose(
                input,
                filters,
                output_shape,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
        # conv3d_transpose
        elif num_spatial_dims == 3:
            output = tf_frontend.nn.conv3d_transpose(
                input,
                filters,
                output_shape,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
        output = ivy.permute_dims(
            output, axes=[0, num_spatial_dims + 1] + [*range(1, num_spatial_dims + 1)]
        )
    else:
        # conv1d
        if num_spatial_dims == 1:
            output = tf_frontend.nn.conv1d(
                input,
                filters,
                strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
        # atrous_conv2d
        elif num_spatial_dims == 2 and is_atrous_conv:
            output = tf_frontend.nn.atrous_conv2d(input, filters, dilations, padding)
        # conv2d
        elif num_spatial_dims == 2 and not is_atrous_conv:
            output = tf_frontend.nn.conv2d(
                input,
                filters,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
        # conv3d
        elif num_spatial_dims == 3:
            output = tf_frontend.nn.conv3d(
                input,
                filters,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
            )
    return output
