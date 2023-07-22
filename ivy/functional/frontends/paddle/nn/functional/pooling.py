# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.func_wrapper import with_supported_dtypes

from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
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


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def avg_pool3d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    /,
    *,
    ceil_mode=False,
    divisor_override=None,
    data_format="NDHWC",
    out=None,
):
    # Dilation must always follows the formuala
    # H(out) = H(in) + 2*Padding[0] - dilation[0]*(kernel_size[0]-1)
    #          -----------------------------------------------------   + 1
    #                           Stride[0]
    # check_neg_arr(x, kernel_size, padding, stride, dilation, data_format)

    return ivy.avg_pool3d(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        data_format=data_format,
        out=out,
    )
