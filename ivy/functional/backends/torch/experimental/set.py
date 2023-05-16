# global
import torch
from typing import Optional


def union(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    res = set(x1+x2)
    return res


union.support_native_out = True


