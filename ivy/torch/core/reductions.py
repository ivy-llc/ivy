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
    return _torch.sum(x, dim=axis, keepdim=keepdims)


def reduce_prod(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return _torch.prod(x, dim=axis, keepdim=keepdims)
    for i, a in enumerate(axis):
        x = _torch.prod(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x


def reduce_mean(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    return _torch.mean(x, dim=axis, keepdim=keepdims)


def reduce_min(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return _torch.min(x, dim=axis, keepdim=keepdims).values
    for i, a in enumerate(axis):
        x = _torch.min(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    return x


def reduce_max(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return _torch.max(x, dim=axis, keepdim=keepdims).values
    for i, a in enumerate(axis):
        x = _torch.max(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    return x
