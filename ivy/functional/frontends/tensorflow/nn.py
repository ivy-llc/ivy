# global
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.tensorflow import math


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
