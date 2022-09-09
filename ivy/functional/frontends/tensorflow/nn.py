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


conv2d.unsupported_dtypes = {"torch": ("float16",)}


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


conv2d_transpose.unsupported_dtypes = {"torch": ("float16",)}


def conv3d(
    input, filters, strides, padding, data_format="NDHWC", dilations=None, name=None
):
    return ivy.conv3d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


conv3d.unsupported_dtypes = {"torch": ("float16",)}


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


conv3d_transpose.unsupported_dtypes = {"torch": ("float16",)}


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
    inv = 1.0 / ivy.sqrt(variance + variance_epsilon)
    if scale is not None:
        inv *= scale

    return x * ivy.astype(inv, x.dtype, copy=False) + ivy.astype(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype
    )
