# global
import paddle
from typing import Union, Optional, Sequence
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def all(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x = x.cast("bool")
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        return paddle.all(x, axis=axis, keepdim=keepdims)
    axis = [i % x.ndim for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = paddle.all(x, axis=a if keepdims else a - i, keepdim=keepdims)
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def any(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x = x.cast("bool")
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        return paddle.any(x, axis=axis, keepdim=keepdims)
    axis = [i % x.ndim for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = paddle.any(x, axis=a if keepdims else a - i, keepdim=keepdims)
    return x
