# global
import paddle
from typing import Union, Optional, Sequence
import ivy.functional.backends.paddle as paddle_backend


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
        ret = paddle.all(x, axis=axis, keepdim=keepdims)
    else:
        axis = [i % x.ndim for i in axis]
        axis.sort()
        ret = x.clone()
        for i, a in enumerate(axis):
            ret = paddle.all(ret, axis=a if keepdims else a - i, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = paddle_backend.squeeze(ret, axis=-1)
    return ret


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
        ret = paddle.any(x, axis=axis, keepdim=keepdims)
    else:
        axis = [i % x.ndim for i in axis]
        axis.sort()
        ret = x.clone()
        for i, a in enumerate(axis):
            ret = paddle.any(ret, axis=a if keepdims else a - i, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = paddle_backend.squeeze(ret, axis=-1)
    return ret
