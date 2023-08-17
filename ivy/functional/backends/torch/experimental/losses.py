from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def l1_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.l1_loss(
        x, target, size_average=size_average, reduce=reduce, reduction=reduction
    )
