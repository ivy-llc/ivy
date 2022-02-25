# global
import math
import torch

def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)

def sqrt(x: torch.Tensor)\
    -> torch.Tensor:
    if isinstance(x, float):
        return math.sqrt(x)
    return torch.sqrt(x)
