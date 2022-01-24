"""
Collection of PyTorch reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch
from typing import Optional, List


def reduce_sum(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    ret = _torch.sum(x, dim=axis, keepdim=keepdims)
    if ret.shape == ():
        return ret.view((1,))
    return ret


def reduce_prod(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        ret = _torch.prod(x, dim=axis, keepdim=keepdims)
        if ret.shape == ():
            return ret.view((1,))
        return ret
    dims = len(x.shape)
    axis = [i%dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = _torch.prod(x, dim=a if keepdims else a - i, keepdim=keepdims)
    ret = x
    if ret.shape == ():
        return ret.view((1,))
    return ret


def reduce_mean(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    ret = _torch.mean(x, dim=axis, keepdim=keepdims)
    if ret.shape == ():
        return ret.view((1,))
    return ret


def reduce_var(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    ret = _torch.var(x, dim=axis, unbiased=False, keepdim=keepdims)
    if ret.shape == ():
        return ret.view((1,))
    return ret


def reduce_min(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        ret = _torch.min(x, dim=axis, keepdim=keepdims).values
        if ret.shape == ():
            return ret.view((1,))
        return ret
    dims = len(x.shape)
    axis = [i%dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = _torch.min(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    ret = x
    if ret.shape == ():
        return ret.view((1,))
    return ret


def reduce_max(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        ret = _torch.max(x, dim=axis, keepdim=keepdims).values
        if ret.shape == ():
            return ret.view((1,))
        return ret
    dims = len(x.shape)
    axis = [i%dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = _torch.max(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    ret = x
    if ret.shape == ():
        return ret.view((1,))
    return ret


def einsum(equation, *operands):
    ret = _torch.einsum(equation, *operands)
    if ret.shape == ():
        return ret.view((1,))
    return ret
