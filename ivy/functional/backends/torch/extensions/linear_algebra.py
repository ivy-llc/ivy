# global
import torch
from typing import Optional


def diagflat(
    x: torch.Tensor,
    /,
    *,
    k: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    return torch.diagflat(x, offset=k)
