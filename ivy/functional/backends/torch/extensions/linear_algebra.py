# global
from typing import Optional

import torch


def diagflat(
    x: torch.Tensor,
    /,
    *,
    k: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    return torch.diagflat(x, offset=k)
