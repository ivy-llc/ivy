# global
import torch
import math


def bitwise_and(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(x1, x2)


def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)


def cos(x: torch.Tensor)\
        -> torch.Tensor:
    if isinstance(x, float):
        return math.cos(x)
    return torch.cos(x)


def logical_not(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool))
