# global
from functools import reduce

# local
import ivy
from ivy import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


# --- Helpers --- #
# --------------- #


def _broadcast_pooling_helper(x, pool_dims: str = "2d", name: str = "padding"):
    dims = {"1d": 1, "2d": 2, "3d": 3}

    if isinstance(x, int):
        return tuple(x for _ in range(dims[pool_dims]))

    if len(x) == 1:
        return tuple(x[0] for _ in range(dims[pool_dims]))
    elif len(x) == dims[pool_dims]:
        return tuple(x)
    else:
        raise ValueError(
            f"`{name}` must either be a single int, "
            f"or a tuple of {dims[pool_dims]} ints. "
        )


# --- Main --- #
# ------------ #


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
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
        "2.1.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):
    return ivy.adaptive_avg_pool2d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
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
    {"2.1.0 and below": ("float16",)},
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
    return ivy.avg_pool1d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        [(pad, pad) for pad in padding],
        data_format="NCW",
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16",)},
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
    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        [(pad, pad) for pad in padding],
        data_format="NCHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16")},
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
    return ivy.avg_pool3d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        [(pad, pad) for pad in padding],
        data_format="NCDHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
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


@to_ivy_arrays_and_back
def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    dilation=1,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    data_format = "NCW"
    return ivy.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, "torch")
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
    return ivy.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, "torch")
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

    return ivy.max_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCDHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
