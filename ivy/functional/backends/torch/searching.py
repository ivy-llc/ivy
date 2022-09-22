from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as tnf

import ivy


# Array API Standard #
# ------------------ #


def argmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
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
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = torch.tensor(x)
    return torch.argmin(x, axis=axis, keepdim=keepdims, out=out)


argmin.support_native_out = True


def nonzero(
    x: torch.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    res = torch.stack(torch.nonzero(x, as_tuple=True))
    if size is not None:
        diff = size - res[0].shape[0]
        if diff > 0:
            res = tnf.pad(res, (0, diff), value=fill_value)
        elif diff < 0:
            res = res[:, :size]

    res = tuple(res)
    if as_tuple:
        return res
    return torch.stack(res, dim=1)


def where(
    condition: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.where(condition, x1, x2).to(dtype=x1.dtype)


# Extra #
# ----- #


def argwhere(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.argwhere(x)
