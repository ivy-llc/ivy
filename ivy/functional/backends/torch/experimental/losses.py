from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    backend_version,
)
def hinge_embedding_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    margin: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.hinge_embedding_loss(
        input,
        target,
        margin=margin,
        reduction=reduction,
        out=out,
    )
