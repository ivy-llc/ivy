# global
import torch


def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)


def logical_not(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool))
