from numbers import Number
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as tnf

import ivy

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Array API Standard #
# ------------------ #


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def argmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if select_last_index:
        if axis is None:
            x = torch.flip(x, dims=[axes for axes in range(x.ndim)])
            ret = torch.argmax(x, dim=axis, keepdim=keepdims)
            ret = x.numel() - ret - 1
        else:
            x = torch.flip(x, dims=(axis,))
            ret = torch.argmax(x, dim=axis, keepdim=keepdims)
            ret = x.shape[axis] - ret - 1
    else:
        ret = torch.argmax(x, dim=axis, keepdim=keepdims)
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.to(dtype=dtype)
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def argmin(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,
    select_last_index: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if select_last_index:
        if axis is None:
            x = torch.flip(x, dims=[axes for axes in range(x.ndim)])
            ret = torch.argmin(x, dim=axis, keepdim=keepdims)
            ret = x.numel() - ret - 1
        else:
            x = torch.flip(x, dims=(axis,))
            ret = torch.argmin(x, dim=axis, keepdim=keepdims)
            ret = x.shape[axis] - ret - 1
    else:
        ret = torch.argmin(x, dim=axis, keepdim=keepdims)
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.to(dtype=dtype)
    return ret


def nonzero(
    x: torch.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    res = torch.stack(torch.nonzero(x, as_tuple=True))

    if size is not None:
        if isinstance(fill_value, float):
            res = res.to(dtype=torch.float64)

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
    x1: Union[float, int, torch.Tensor],
    x2: Union[float, int, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if condition.dtype is not torch.bool:
        condition = condition == 1.0
    return ivy.astype(torch.where(condition, x1, x2), x1.dtype, copy=False)


# Extra #
# ----- #


def argwhere(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.argwhere(x)
