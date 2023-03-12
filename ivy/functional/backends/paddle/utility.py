# global
import paddle
from typing import Union, Optional, Sequence
from ivy.utils.exceptions import IvyNotImplementedException


def all(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x = paddle.to_tensor(x, dtype=paddle.bool)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return paddle.all(x, axis=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = paddle.all(x, axis=a if keepdims else a - i, keepdim=keepdims)
    return x


def any(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x = paddle.to_tensor(x, dtype=paddle.bool)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return paddle.any(x, axis=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = paddle.any(x, axis=a if keepdims else a - i, keepdim=keepdims)
    return x
