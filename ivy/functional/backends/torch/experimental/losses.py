from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes(
    {"2.0.1 and below": ("unit8", "int8", "int16", "int32", "int64", "bool")},
    backend_version,
)
def l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: Optional[str] = "mean",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.l1_loss(
        input,
        target,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        out=out,
    )
