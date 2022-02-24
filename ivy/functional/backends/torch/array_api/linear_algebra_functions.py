import torch
from typing import Union, Optional, Tuple, List


def cross(x1: torch.Tensor, x2: torch.Tensor, /, *, axis: int = -1) -> torch.Tensor:
    return torch.cross(a=x1, b=x2, dim=axis)
