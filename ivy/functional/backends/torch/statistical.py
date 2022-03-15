# global
torch_scatter = None
import torch
from typing import Tuple, Union


def min(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return torch.amin(input = x)
    return torch.amin(input = x, dim = axis, keepdim = keepdims)

def max(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return torch.amax(input = x)
    return torch.amax(input = x, dim = axis, keepdim = keepdims)