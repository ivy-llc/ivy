import torch
from typing import Union, Optional, Tuple, List


def var(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> torch.Tensor:

    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return torch.var(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = torch.var(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x
