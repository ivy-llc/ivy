# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
import numpy as np
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.nn.functional.pooling_functions import (
    _broadcast_pooling_helper,
)
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool1d(x, output_size, name=None):
    return ivy.adaptive_avg_pool1d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool2d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool3d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool3d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_max_pool2d(x, output_size, return_mask=None, name=None):
    return ivy.adaptive_max_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
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
        [pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)]
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
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    # Figure out padding string
    if all(
        [pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)]
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


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    # Check the shapes of the input and kernel tensors.
    # Ensure input is a NumPy array.
    if not isinstance(input, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Check the shape of the input tensor.
    if input.shape != (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
        input.shape[4],
    ):
        raise ValueError(
            "Input tensor must have shape (batch, channel, depth, height, width)."
        )

    # Validate and normalize parameter shapes.
    def validate_param(param_name, param_value):
        if isinstance(param_value, int):
            return (param_value, param_value, param_value)
        elif isinstance(param_value, (tuple, list)) and len(param_value) == 3:
            return tuple(int(val) for val in param_value)
        else:
            raise ValueError(
                f"{param_name} must be an int or a tuple/list of three integers."
            )

    kernel_size = validate_param("kernel_size", kernel_size)
    stride = validate_param("stride", stride) if stride is not None else kernel_size
    padding = validate_param("padding", padding)
    dilation = validate_param("dilation", dilation)

    # Create a 3D max pooling operation.
    method = "ceil" if ceil_mode else "floor"
    return ivy.max_pool3d(input, kernel_size, stride, padding, dilation, method)


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
