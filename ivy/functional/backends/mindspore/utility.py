# global
import mindspore as ms
import mindspore.ops as ops
from typing import Union, Optional, Sequence

def all(
    x: ms.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = x.astype(dtype=ms.bool_)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return ops.all(x, axis=axis, keep_dims=keepdims)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = ops.all(x, axis=a if keepdims else a - i, keep_dims=keepdims)
    return x
