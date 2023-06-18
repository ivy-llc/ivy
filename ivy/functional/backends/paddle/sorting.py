# global
import paddle
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
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
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x = x.cast("float32")
    return paddle.argsort(x, axis=axis, descending=descending)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
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
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        return paddle.sort(x.cast("float32"), axis=axis, descending=descending).cast(
            x.dtype
        )
    return paddle.sort(x, axis=axis, descending=descending)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
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
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x = x.cast("float32")

    if v.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        v = v.cast("float32")

    right = True if side == "right" else False
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )

    if sorter is not None:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
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
    {"2.4.2 and below": {"cpu": ("int8", "uint8", "int16", "float16", "complex")}},
    backend_version,
)
def msort(
    a: Union[paddle.Tensor, list, tuple], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.sort(a, axis=0)
