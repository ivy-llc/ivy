# global
import ivy
import torch
from typing import Union, Optional, Tuple, List


# noinspection PyShadowingBuiltins
def all(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x.type(torch.bool)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return torch.all(x, dim=axis, keepdim=keepdims, out=out)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = torch.all(x, dim=a if keepdims else a - i, keepdim=keepdims, out=out)
    return x


# noinspection PyShadowingBuiltins
def any(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = ivy.asarray(x).type(torch.bool)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return torch.any(x, dim=axis, keepdim=keepdims, out=out)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = torch.any(x, dim=a if keepdims else a - i, keepdim=keepdims, out=out)
    return x
