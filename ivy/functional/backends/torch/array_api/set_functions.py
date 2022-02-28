# global
import torch
from torch import Tensor
from typing import Tuple
from collections import namedtuple

def unique_counts(x: Tensor) \
                -> Tuple[Tensor, Tensor]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, c = torch.unique(torch.reshape(x, [-1]), return_counts=True)
    return uc(v, c)