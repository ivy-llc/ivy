# global
import torch
from typing import Optional

# local
import ivy


def argsort(
    x: torch.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        out = tuple([torch.zeros(x.shape, dtype=x.dtype), out.long()])
    _, sorted_indices = torch.sort(
        x, dim=axis, descending=descending, stable=stable, out=out
    )
    return sorted_indices


argsort.support_native_out = True


def sort(
    x: torch.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        out = tuple([out, torch.zeros(out.shape, dtype=torch.long)])
    sorted_tensor, _ = torch.sort(
        x, dim=axis, descending=descending, stable=stable, out=out
    )
    return sorted_tensor


sort.support_native_out = True


def searchsorted(
    x: torch.Tensor,
    v: torch.Tensor,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=torch.int64,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(ret_dtype)
    if dtype is torch.int64:
        ret_int32 = False
    elif dtype is torch.int32:
        ret_int32 = True
    else:
        raise ValueError("only int32 and int64 are supported for ret_dtype.")
    return torch.searchsorted(
        x,
        v,
        sorter=sorter,
        side=side,
        out_int32=ret_int32,
        out=out,
    )


searchsorted.support_native_out = True
