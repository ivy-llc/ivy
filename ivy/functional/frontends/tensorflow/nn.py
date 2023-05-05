# global
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from ivy.functional.frontends.tensorflow import math


def _reduce_strides_dilations(dim, stride, dilations):
    if not isinstance(stride, int):
        if len(stride) > dim:
            stride = stride[1:-1]
        if len(stride) == 1 and dim != 1:
            stride = stride[0]
    if not isinstance(dilations, int):
        if len(dilations) > dim:
            dilations = dilations[1:-1]
        if len(dilations) == 1 and dim != 1:
            dilations = dilations[0]
    return stride, dilations


@to_ivy_arrays_and_back
def atrous_conv2d(value, filters, rate, padding):
    return ivy.conv2d(value, filters, 1, padding, dilations=[rate] * 2)


@to_ivy_arrays_and_back
def atrous_conv2d_transpose(value, filters, output_shape, rate, padding):
    filters = filters.swapaxes(-2, -1)
    return ivy.conv2d_transpose(
        value, filters, 1, padding, output_shape=output_shape, dilations=[rate] * 2
    )


@to_ivy_arrays_and_back
def conv1d(
    input, filters, stride, padding, data_format="NWC", dilations=None, name=None
):
    dilations = 1 if dilations is None else dilations
    stride, dilations = _reduce_strides_dilations(1, stride, dilations)
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
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(1, strides, dilations)
    filters = filters.swapaxes(-2, -1)
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
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
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
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    filters = filters.swapaxes(-2, -1)
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
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(3, strides, dilations)
    return ivy.conv3d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


@with_unsupported_dtypes({"2.9.0 and below": ("bfloat16",)}, "tensorflow")
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
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(3, strides, dilations)
    filters = filters.swapaxes(-2, -1)
    return ivy.conv3d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def depthwise_conv2d(
    input,
    filter,
    strides,
    padding="SAME",
    data_format="NHWC",
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    fc = filter.shape[-2]
    filter = filter.reshape(
        [*filter.shape[0:2], 1, filter.shape[-2] * filter.shape[-1]]
    )
    return ivy.conv_general_dilated(
        input,
        filter,
        strides,
        padding,
        data_format="channel_last" if data_format[-1] == "C" else "channel_first",
        dilations=dilations,
        feature_group_count=fc,
    )


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def separable_conv2d(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    ret = depthwise_conv2d(
        input,
        depthwise_filter,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format=data_format,
    )
    return conv2d(ret, pointwise_filter, 1, "SAME", data_format=data_format)


@to_ivy_arrays_and_back
def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
    xnormalized, _, _ = ivy.batch_norm(
        x,
        mean,
        variance,
        offset=offset,
        scale=scale,
        eps=variance_epsilon,
    )
    return xnormalized


@to_ivy_arrays_and_back
def dropout(x, rate, noise_shape=None, seed=None, name=None):
    return ivy.dropout(x, rate, noise_shape=noise_shape, seed=seed)


@with_unsupported_dtypes(
    {
        "2.9.1": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
            "bfloat16",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def silu(features, beta: float = 1.0):
    beta = ivy.astype(ivy.array(beta), ivy.dtype(features))
    return ivy.multiply(features, ivy.sigmoid(ivy.multiply(beta, features)))


@with_unsupported_dtypes(
    {
        "2.9.1": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    ivy.utils.assertions.check_shape(labels, logits)
    zeros = ivy.zeros_like(logits)
    max_logits = ivy.where(logits >= zeros, logits, zeros)
    neg_abs_logits = ivy.negative(ivy.abs(logits))
    neg_multiple = ivy.negative(ivy.multiply(logits, labels))
    ret_val = ivy.add(max_logits, neg_multiple)
    return ivy.add(ret_val, ivy.log1p(ivy.exp(neg_abs_logits)))


@with_unsupported_dtypes(
    {
        "2.9.1": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def weighted_cross_entropy_with_logits(
    labels=None, logits=None, pos_weight=1.0, name=None
):
    ivy.utils.assertions.check_shape(labels, logits)
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


@with_supported_dtypes(
    {"2.9.0 and below": ("float32", "float16", "bfloat16")}, "tensorflow"
)
@to_ivy_arrays_and_back
def local_response_normalization(
    input, /, *, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None
):
    input_shape = ivy.shape(input)
    ivy.utils.assertions.check_equal(
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
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    return ivy.max_pool2d(input, ksize, strides, padding, data_format=data_format)


@to_ivy_arrays_and_back
def moments(x, axes, shift=None, keepdims=False, name=None):
    return ivy.mean(x, axis=ivy.to_list(axes), keepdims=keepdims), ivy.var(
        x, axis=ivy.to_list(axes), keepdims=keepdims
    )


@to_ivy_arrays_and_back
def bias_add(value, bias, data_format=None, name=None):
    if data_format is None:
        data_format = "N...C"

    chanel_index = data_format.find("C")
    if chanel_index != 1:
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
        if isinstance(arg, int):
            arg = [arg]
        else:
            arg = list(arg)
        len_arg = len(arg)

        if len_arg == num_spatial_dims + 2:
            return arg

        # Broadcast to rcorrect dimensions
        if len_arg == 1:
            arg = arg * num_spatial_dims
        elif len_arg != num_spatial_dims:
            raise ValueError(
                f"{name} should be of length 1, "
                f"{num_spatial_dims} or {num_spatial_dims + 2}. "
                f"Received: {name}={arg} of length {len_arg}."
            )

    # Add dimensions for batch and channel
    if channel_index == 1:
        return [1, 1] + arg
    else:
        return [1] + arg + [1]


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
    num_spatial_dims = input.ndim - 2
    if data_format is None or not data_format.startswith("NC"):
        data_format = "channel_last"
    else:
        data_format = "channel_first"

    channel_index = -1 if data_format == "channel_last" else 1
    input_depth = ivy.shape(input)[channel_index]
    filters_depth = ivy.shape(filters)[-2]

    feature_group_count = 1
    if input_depth != filters_depth:
        if input_depth % filters_depth != 0:
            raise ValueError(
                "input depth must be evenly divisible by filter depth: "
                f"{input_depth} vs {filters_depth}"
            )
        feature_group_count = input_depth // filters_depth
    return ivy.conv_general_dilated(
        input,
        filters,
        strides,
        padding,
        dims=num_spatial_dims,
        data_format=data_format,
        dilations=dilations,
        feature_group_count=feature_group_count,
    )


@to_ivy_arrays_and_back
def embedding_lookup(params, ids, max_norm=None, name=None):
    return ivy.embedding(params, ids, max_norm=max_norm)


@to_ivy_arrays_and_back
def relu(features, name=None):
    return ivy.relu(features)


@to_ivy_arrays_and_back
def relu6(features, name=None):
    return ivy.relu6(features)


@to_ivy_arrays_and_back
def softmax(logits, axis=None, name=None):
    return ivy.softmax(logits, axis=axis)


@with_unsupported_dtypes({"2.9.0 and below": "float16"}, "tensorflow")
@to_ivy_arrays_and_back
def leaky_relu(features, alpha, name=None):
    return ivy.leaky_relu(features, alpha=alpha)


@to_ivy_arrays_and_back
def crelu(features, axis=-1, name=None):
    c = ivy.concat([features, -features], axis=axis)
    return ivy.relu(c)


@to_ivy_arrays_and_back
def avg_pool(input, ksize, strides, padding, data_format="NWC", name=None):
    if len(ivy.shape(input)) == 3:
        return ivy.avg_pool1d(input, ksize, strides, padding, data_format=data_format)
    elif len(ivy.shape(input)) == 4:
        return ivy.avg_pool2d(input, ksize, strides, padding, data_format=data_format)
    return ivy.avg_pool3d(input, ksize, strides, padding, data_format=data_format)


@to_ivy_arrays_and_back
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    return ivy.avg_pool3d(input, ksize, strides, padding, data_format=data_format)
