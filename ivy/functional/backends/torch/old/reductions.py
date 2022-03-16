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


def einsum(equation, *operands):
    return _torch.einsum(equation, *operands)