# global
import torch
from typing import Union, Optional, Sequence


def all(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
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


all.support_native_out = True


def any(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = torch.as_tensor(x).type(torch.bool)
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


any.support_native_out = True
