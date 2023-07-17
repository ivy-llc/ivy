# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

def _broadcast_pooling_helper(x, pool_dims: str = "2d", name: str = "padding"):
    dims = {"1d": 1, "2d": 2, "3d": 3}

    if isinstance(x, int):
        return tuple([x for _ in range(dims[pool_dims])])

    if len(x) == 1:
        return tuple([x[0] for _ in range(dims[pool_dims])])
    elif len(x) == dims[pool_dims]:
        return tuple(x)
    elif len(x) != dims[pool_dims]:
        raise ValueError(
            f"`{name}` must either be a single int, "
            f"or a tuple of {dims[pool_dims]} ints. "
        )


@with_supported_device_and_dtypes(
    {
        "2.5.0 and above": {
            "cpu": (
                "bfloat16",
                "float32",
                "float64",
            ),
            "gpu": (
                "bfloat16",
                "float16",
                "float32",
                "float64",
            ),
        },
        "2.4.2 and below": {
            "cpu": (
                "float32",
                "float64",
            ),
            "gpu": (
                "float16",
                "float32",
                "float64",
            ),
        },
    },
    "paddle",
)
@to_ivy_arrays_and_back
def max_pool1d(x, kernel_size, stride=None, padding=0):
    kernel_size = _broadcast_pooling_helper(kernel_size, "1d", name="kernel_size")
    stride = _broadcast_pooling_helper(stride, "1d", name="stride")
    padding = _broadcast_pooling_helper(padding, "1d", name="padding")
    kernel_pads = list(zip(kernel_size, padding))

    data_format = "NCW"

    if not all([pad <= kernel / 2 for kernel, pad in kernel_pads]):
        raise ValueError(
            "pad should be smaller than or equal to half of kernel size, "
            f"but got padding={padding}, kernel_size={kernel_size}. "
        )
    # figure out whether to apply padding
    if all([pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in kernel_pads]):
        padding_str = "SAME"
    else:
        padding_str = "VALID"

    return ivy.max_pool1d(x, kernel_size, stride, padding_str, data_format=data_format)
