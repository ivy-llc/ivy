from typing import Optional, Tuple

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
    ret = torch.argmax(x, dim=axis, keepdim=keepdims, out=out)
    return ret


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
    ret = torch.argmin(x, axis=axis, keepdim=keepdims, out=out)
    return ret


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
    promoted_type = torch.promote_types(x1.dtype, x2.dtype)
    x1 = x1.to(promoted_type)
    x2 = x2.to(promoted_type)
    return torch.where(condition, x1, x2)
