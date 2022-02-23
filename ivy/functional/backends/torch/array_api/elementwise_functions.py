# global
import torch


def isfinite(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.isfinite(x)


def cosh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.cosh(x)
