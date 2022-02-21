



# global
import math
import torch as torch


def cos(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, float):
        return math.cos(x)
    return torch.cos(x)
