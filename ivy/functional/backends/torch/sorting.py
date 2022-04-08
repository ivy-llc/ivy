# global
import torch
from typing import Optional

import ivy


def argsort(x: torch.Tensor,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True,
            out: Optional[torch.Tensor] = None)\
        -> torch.Tensor:
    ret = torch.argsort(x, dim=axis, descending=descending)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sort(x: torch.Tensor,
         axis: int = -1,
         descending: bool = False,
         stable: bool = True,
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    sorted_tensor, _ = torch.sort(x, dim=axis, descending=descending)
    if ivy.exists(out):
        return ivy.inplace_update(out, sorted_tensor)
    return sorted_tensor
