import torch
from typing import Union, Optional, Tuple, List


def var(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.var(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        x = torch.var(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x


def min(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return torch.amin(input = x)
    return torch.amin(input = x, dim = axis, keepdim = keepdims)
