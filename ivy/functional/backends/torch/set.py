# global
from typing import Tuple
import torch


def unique_inverse(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    values, inverse_indices = torch.unique(x, return_inverse=True)
    return values, inverse_indices
