# global
import torch
from typing import Tuple
from collections import namedtuple

def unique_inverse(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = torch.unique(x, return_inverse=True)
    return out(values, inverse_indices)
