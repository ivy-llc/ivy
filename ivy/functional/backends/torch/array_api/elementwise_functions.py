# global
import torch
import math

def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)

def tanh(x: torch.Tensor)\
        -> torch.Tensor:
        if isinstance(x, float):
            return math.tanh(x)
        return torch.tanh(x)