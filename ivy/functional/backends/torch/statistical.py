# global
torch_scatter = None
import torch as _torch
from typing import Tuple, Union, Optional


def min(x: _torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return _torch.amin(input = x)
    return _torch.amin(input = x, dim = axis, keepdim = keepdims)

def max(x: _torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return _torch.amax(input = x)
    return _torch.amax(input = x, dim = axis, keepdim = keepdims)


def var(x: _torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> _torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return _torch.var(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        x = _torch.var(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x
