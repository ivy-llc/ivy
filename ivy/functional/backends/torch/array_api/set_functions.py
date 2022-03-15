# global
from typing import Tuple
import torch as th


def unique_inverse(x: th.Tensor) \
        -> Tuple[th.Tensor, th.Tensor]:
    values, inverse_indices = th.unique(x, return_inverse=True)
    return values, inverse_indices
