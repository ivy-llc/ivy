import torch
from typing import Optional, List

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, backend_version)
def layer_norm(
    x: torch.Tensor,
    normalized_idxs: List[int],
    /,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    new_std: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    normalized_shape = x.shape[normalized_idxs[0] :]
    xnormalized = torch.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=offset, eps=eps
    )
    return torch.multiply(xnormalized, new_std)
