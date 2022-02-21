# global
import torch
import math


def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)


def cos(x: torch.Tensor)\
        -> torch.Tensor:
    if isinstance(x, float):
        return math.cos(x)
    return torch.cos(x)
