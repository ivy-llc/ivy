# global
import ivy


def atrous_conv2d(value, filters, rate, padding):
    return ivy.conv2d(value, filters, 1, padding, dilations=rate)


def atrous_conv2d_transpose(value, filters, output_shape, rate, padding):
    return ivy.conv2d_transpose(
        value, filters, rate, padding, output_shape=output_shape, dilations=rate
    )


def conv1d(
    input, filters, stride, padding, data_format="NWC", dilations=None, name=None
):
    return ivy.conv1d(
        input, filters, stride, padding, data_format=data_format, dilations=dilations
    )


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


def gelu(features, approximate=False, name=None):
    return ivy.gelu(features, approximate=approximate)


def conv2d(
    input, filters, strides, padding, data_format="NHWC", dilations=None, name=None
):
    return ivy.conv2d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


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


def conv3d(
    input, filters, strides, padding, data_format="NDHWC", dilations=None, name=None
):
    return ivy.conv3d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


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


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
    inv = 1.0 / ivy.sqrt(variance + variance_epsilon)
    if scale is not None:
        inv *= scale

    return x * ivy.astype(inv, x.dtype, copy=False) + ivy.astype(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype
    )


def dropout(x, prob, scale, dtype, name=None):
    return ivy.dropout(x, prob, scale, dtype)


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
