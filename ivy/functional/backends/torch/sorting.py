# global
import torch
from typing import Optional


def argsort(
    x: torch.Tensor, axis: int = -1, descending: bool = False, stable: bool = True
) -> torch.Tensor:
    ret = torch.argsort(x, dim=axis, descending=descending)
    return ret


def sort(
    x: torch.Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sorted_tensor, _ = torch.sort(x, dim=axis, descending=descending, out=out)
    return sorted_tensor
