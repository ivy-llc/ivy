# global
import paddle
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes, with_supported_dtypes
from . import backend_version


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def argsort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.argsort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def sort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.sort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def searchsorted(
    x: paddle.Tensor,
    v: paddle.Tensor,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=paddle.int64,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    right = True if side == "right" else False
    assert ivy.is_int_dtype(ret_dtype), TypeError(
        "only Integer data types are supported for ret_dtype."
    )

    if sorter is not None:
        assert ivy.is_int_dtype(sorter.dtype), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
        if ivy.as_native_dtype(sorter.dtype) not in [paddle.int32, paddle.int64]:
            sorter = sorter.cast(paddle.int64)
        x = paddle.take_along_axis(x, sorter, axis=-1)

    if x.ndim != 1:
        assert x.shape[:-1] == v.shape[:-1], RuntimeError(
            "the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )

    return paddle.searchsorted(x, v, right=right).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("int8", "uint8", "int16", "float16", "bfloat16", "complex")
        }
    },
    backend_version,
)
def msort(
    a: Union[paddle.Tensor, list, tuple], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.sort(a, axis=0)
