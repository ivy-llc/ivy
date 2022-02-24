# global
import torch
import math

def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)


def atanh(x: torch.Tensor)\
        -> torch.Tensor:
    if isinstance(x, float):
        return math.atanh(x)
    return torch.atanh(x)
