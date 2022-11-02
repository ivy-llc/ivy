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


def kron(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    return torch.kron(a, b, out=out)


kron.support_native_out = True
