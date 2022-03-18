"""
Collection of PyTorch linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch
from typing import List
import ivy as _ivy
from typing import Union, Tuple


def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = _torch.unsqueeze(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = _torch.zeros(batch_shape + [1, 1], device=vector.device)
    # BS x 1 x 3
    row1 = _torch.cat((zs, -a3s, a2s), -1)
    row2 = _torch.cat((a3s, zs, -a1s), -1)
    row3 = _torch.cat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return _torch.cat((row1, row2, row3), -2)
