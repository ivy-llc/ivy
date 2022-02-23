# global
import torch
from typing import Union, Optional, Tuple, List


# noinspection PyShadowingBuiltins
def all(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False)\
        -> torch.Tensor:
    x = x.type(torch.bool)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return torch.all(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = [i%dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = torch.all(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x
