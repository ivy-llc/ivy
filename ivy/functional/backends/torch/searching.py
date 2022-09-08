from typing import Optional, Tuple

import ivy
import torch


def argmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = torch.tensor(x)
    return torch.argmax(x, dim=axis, keepdim=keepdims, out=out)


argmax.support_native_out = True


def argmin(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = torch.tensor(x)
    return torch.argmin(x, axis=axis, keepdim=keepdims, out=out)


argmin.support_native_out = True


def nonzero(
    x: torch.Tensor,
    /,
) -> Tuple[torch.Tensor]:
    return torch.nonzero(x, as_tuple=True)


def where(
    condition: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.where(condition, x1, x2)
