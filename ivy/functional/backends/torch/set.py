# global
import torch
from typing import Tuple
from collections import namedtuple

import ivy

def unique_inverse(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = torch.unique(x, return_inverse=True)
    nan_idx = torch.isnan(x)
    if nan_idx.any():
        inverse_indices[nan_idx] = torch.where(torch.isnan(values))[0][0]
    inverse_indices = inverse_indices.reshape(x.shape)
    return out(values, inverse_indices)


def unique_values(x: torch.Tensor, out: torch.Tensor = None) \
        -> torch.Tensor:
    ret = torch.unique(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def unique_counts(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    v, c = torch.unique(torch.reshape(x, [-1]), return_counts=True)
    nan_idx = torch.where(torch.isnan(v))
    c[nan_idx] = 1
    uc = namedtuple('uc', ['values', 'counts'])
    return uc(v, c)
