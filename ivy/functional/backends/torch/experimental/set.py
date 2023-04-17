# global
import torch
from typing import Optional
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


def difference(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return x2[(x2[:, None] != x1).all(dim=1)]


difference.support_native_out = True
