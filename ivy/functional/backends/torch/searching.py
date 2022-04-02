import torch
from typing import Optional, Union

def argmax(
    x: torch.Tensor,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    keepdims: Optional[bool] = False
) -> torch.Tensor:
    x = torch.tensor(x)
    ret = torch.argmax(x, dim=axis, out=out, keepdim=keepdims)
    return ret


def argmin(
    x: torch.Tensor,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False)\
    -> torch.Tensor:

    x = torch.tensor(x)
    ret = torch.argmin(x, axis=axis, keepdim=keepdims)
    return ret


def where(condition: torch.Tensor,
          x1: torch.Tensor,
          x2: torch.Tensor)\
        -> torch.Tensor:
    promoted_type = torch.promote_types(x1.dtype, x2.dtype)
    x1 = x1.to(promoted_type)
    x2 = x2.to(promoted_type)
    return torch.where(condition, x1, x2)
