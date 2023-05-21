# global
import torch
from typing import Optional

# local
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex64", "complex128")}, backend_version
)
def difference(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # remove duplicates from x1 and x2
    x1 = torch.unique(x1)
    x2 = torch.unique(x2)
    concatenated = torch.cat([x1, x2])
    unique, counts = torch.unique(concatenated, return_counts=True)
    difference = unique[counts == 1]
    if x1.dtype != torch.bool:
        mask = torch.isin(difference, x1)
        return difference[mask]
    else:
        # convert to numeric
        x1 = x1.float()
        difference = difference.float()
        mask = torch.isin(difference, x1)
        return difference[mask].bool()
