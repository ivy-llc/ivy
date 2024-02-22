# global
from functools import reduce

# local
import ivy
from ivy import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool1d(input, output_size):
    return ivy.adaptive_avg_pool1d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):
    return ivy.adaptive_avg_pool2d(input, output_size, data_format="NCHW")


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_max_pool2d(
    input,
    output_size,
    return_indices=False,
):
    # ToDo: Add return_indices once superset is implemented
    return ivy.adaptive_max_pool2d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "int8",
            "int16",
            "bool",
            "uint8",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_max_pool3d(
    input,
    output_size,
    return_indices=False,
):
    return ivy.adaptive_max_pool3d(input, output_size)


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    "torch",
)
@to_ivy_arrays_and_back
def avg_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool1d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCW",
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    "torch",
)
@to_ivy_arrays_and_back
def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16")},
    "torch",
)
@to_ivy_arrays_and_back
def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool3d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCDHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size

    out = ivy.avg_pool1d(
        ivy.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    p = 1.0 / norm_type if norm_type != 0 else 1.0
    return ivy.pow(ivy.multiply(out, kernel_mul), p)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCHW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    out = ivy.avg_pool2d(
        ivy.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size
    p = ivy.divide(1.0, norm_type) if norm_type != 0 else 1.0
    return ivy.pow(ivy.multiply(out, kernel_mul), p).astype(input.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.max_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCDHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
