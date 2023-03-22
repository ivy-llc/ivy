# global
import paddle
from typing import Optional

# local
import ivy

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def argsort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if ivy.as_native_dtype(x.dtype) == paddle.bool:
        x = x.cast('int32')
    return paddle.argsort(x, axis=axis , descending=descending)


def sort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    
    is_bool = ivy.as_native_dtype(x.dtype) == paddle.bool
    if is_bool:
        x = x.cast('int32')

    ret = paddle.sort(x, axis=axis , descending=descending)

    if is_bool:
        ret = ret.cast('bool')
    
    return ret


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
            f"the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )

    return paddle.searchsorted(x, v, right=right).cast(ret_dtype)

