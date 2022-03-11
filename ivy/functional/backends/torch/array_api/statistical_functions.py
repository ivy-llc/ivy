# global
torch_scatter = None
import torch as _torch
from torch import dtype
from typing import Tuple, Union


def min(x: _torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return _torch.amin(input = x)
    return _torch.amin(input = x, dim = axis, keepdim = keepdims)


def prod(x: _torch.Tensor,
         axis: Union[int, Tuple[int]] = None,
         dtype: dtype = None,
         keepdims: bool = False)\
        -> _torch.Tensor:
    return _torch.prod(x,axis,dtype,keepdims)