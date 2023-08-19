from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    /,
    *,
    full: Optional[bool] = False,
    eps: Optional[float] = 1e-06,
    reduction: Optional[str] = "mean",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.gaussian_nll_loss(
        input,
        target,
        var,
        full=full,
        eps=eps,
        reduction=reduction,
        out=out,
    )
