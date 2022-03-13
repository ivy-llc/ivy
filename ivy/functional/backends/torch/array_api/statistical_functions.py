# global
torch_scatter = None
import torch as _torch
from typing import Tuple, Union, Optional
import numpy as _np

def min(x: _torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return _torch.amin(input = x)
    return _torch.amin(input = x, dim = axis, keepdim = keepdims)


def prod(x: _torch.Tensor,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[_torch.dtype] = None,
         keepdims: bool = False)\
        -> _torch.Tensor:

    if dtype == None:
        if x.dtype == _torch.int8:
            dtype = _torch.int32
        elif x.dtype == _torch.uint8:
            dtype = _np.uint32
        elif x.dtype in [_torch.int64,_torch.int32]: 
            dtype = _torch.int64

    if axis == None:
        axis = 0
    return _torch.prod(input=x,dim=axis,dtype=dtype,keepdim=keepdims)