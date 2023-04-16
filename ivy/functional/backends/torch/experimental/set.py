# global
import torch
from typing import Optional
from ivy.func_wrapper import with_supported_dtypes
from .. import backend_version


@with_supported_dtypes(
    {"2.0.0 and below": ("float16", "float32", "float64")},
    backend_version,
)
def difference(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return x1[(x1[:, None] != x2).all(dim=1)]


difference.support_native_out = True
