import torch
from torch import Tensor
from typing import Union, Tuple, Optional, List


def var(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> torch.Tensor:

    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)

    return torch.var(x, dim=axis, keepdim=keepdims)
