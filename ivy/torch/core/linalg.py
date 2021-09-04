"""
Collection of PyTorch linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch
from typing import Optional, List, Union


# noinspection PyPep8Naming
def svd(x) -> List[torch.Tensor]:
    U, D, V = torch.svd(x)
    VT = torch.transpose(V, -2, -1)
    return U, D, VT


# noinspection PyShadowingBuiltins
def norm(x, ord: Optional[Union[int, str]] = None, axis: Optional[int] = None, keepdims: bool = False):
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def inv(x):
    return torch.inverse(x)


def pinv(x):
    return torch.pinverse(x)


def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = torch.unsqueeze(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = torch.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = torch.cat((zs, -a3s, a2s), -1)
    row2 = torch.cat((a3s, zs, -a1s), -1)
    row3 = torch.cat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return torch.cat((row1, row2, row3), -2)
