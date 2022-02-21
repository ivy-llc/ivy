# global
import torch

def isfinite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)
