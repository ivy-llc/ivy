# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.nn.functional.pooling_functions import (
    _broadcast_pooling_helper,
)
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
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
    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    stride = _broadcast_pooling_helper(stride, "2d", name="stride")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    kernel_pads = list(zip(kernel_size, padding))

    # Padding should be less than or equal to half of kernel size
    if not all([pad <= kernel / 2 for kernel, pad in kernel_pads]):
        raise ValueError(
            "pad should be smaller than or equal to half of kernel size, "
            f"but got padding={padding}, kernel_size={kernel_size}. "
        )

    # Figure out padding string
    if all([pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in kernel_pads]):
        padding = "SAME"
    else:
        padding = "VALID"

    count_include_pad = False if exclusive else True
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
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
def avg_pool1d(
    x, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None
):
    data_format = "NCL"
    exclusive = not exclusive

    return ivy.avg_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        count_include_pad=exclusive,
        ceil_mode=ceil_mode,
        data_format=data_format,
    )

@ivy.to_ivy_arrays_and_back
@ivy.with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_max_pool2d(x, output_size):
    return paddle.nn.functional.adaptive_max_pool2d(x, output_size=output_size)

@ivy.to_ivy_arrays_and_back
@ivy.with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_max_pool3d(x, output_size, name=None):
    return paddle.nn.functional.adaptive_max_pool3d(x, output_size=output_size)

@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool1d(x, output_size, name=None):
    return ivy.adaptive_avg_pool1d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool2d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
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
    return ivy.max_unpool1d(x, indices, kernel_size, stride, padding, data_format)
