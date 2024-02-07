# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.ivy.experimental.layers import _broadcast_pooling_helper
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool1d(x, output_size, name=None):
    return ivy.adaptive_avg_pool1d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool2d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool2d(x, output_size, data_format=data_format)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool3d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool3d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def adaptive_max_pool2d(x, output_size, return_mask=None, name=None):
    return ivy.adaptive_max_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
def avg_pool1d(
    x, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None
):
    data_format = "NCW"
    exclusive = not exclusive
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "1d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "1d", name="padding")
    # Figure out padding string
    if all(
        pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)
    ):
        padding = "SAME"
    else:
        padding = "VALID"

    return ivy.avg_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        count_include_pad=exclusive,
        ceil_mode=ceil_mode,
        data_format=data_format,
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    exclusive=True,
    divisor_override=None,
    data_format="NCHW",
    name=None,
):
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    # Figure out padding string
    if all(
        pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)
    ):
        padding = "SAME"
    else:
        padding = "VALID"

    count_include_pad = not exclusive
    return ivy.avg_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def max_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    return_mask=False,
    ceil_mode=False,
    data_format="NCHW",
    name=None,
):
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            f"Attr(data_format): {data_format}."
        )

    if data_format == "NHWC" and return_mask:
        raise ValueError(
            "When setting return_mask to true, data_format must be set to NCHW in"
            " API:max_pool2d"
        )

    return ivy.max_pool2d(
        x, kernel_size, stride, padding, data_format=data_format, ceil_mode=ceil_mode
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def max_unpool1d(
    x,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    data_format="NCL",
    output_size=None,
    name=None,
):
    return ivy.max_unpool1d(
        x,
        indices,
        kernel_size,
        strides=stride,
        padding=padding,
        data_format=data_format,
    )
