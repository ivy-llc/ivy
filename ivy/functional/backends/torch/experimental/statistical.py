# global
from typing import Optional, Union, Tuple, Sequence
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bool")}, backend_version)
def median(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, tuple):
        if len(axis) == 1:
            axis = axis[0]
    ret = quantile(
        input,
        0.5,
        axis=axis,
        keepdims=keepdims,
        interpolation="midpoint",
    )
    if input.dtype in [torch.int64, torch.float64]:
        ret = torch.asarray(ret, dtype=torch.float64)
    elif input.dtype in [torch.float16, torch.bfloat16]:
        ret = torch.asarray(ret, dtype=input.dtype)
    else:
        ret = torch.asarray(ret, dtype=torch.float32)
    return ret


median.support_native_out = False


def nanmean(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmean(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_unsupported_dtypes(
    {"1.11.0 and below": ("bfloat16", "bfloat32", "float16")}, backend_version
)
def quantile(
    a: torch.Tensor,
    q: Union[torch.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    temp = a.to(torch.float64)
    num_dim = len(temp.size())
    keepdim_shape = list(temp.size())
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(q, torch.Tensor):
        qt = q.to(torch.float64)
    else:
        qt = q
    for i in axis:
        keepdim_shape[i] = 1
    axis = [num_dim + x if x < 0 else x for x in axis]
    axis.sort()
    dimension = len(a.size())
    while len(axis) > 0:
        axis1 = axis[0]
        for axis2 in range(axis1 + 1, dimension):
            temp = torch.transpose(temp, axis1, axis2)
            axis1 = axis2
        axis = [x - 1 for x in axis]
        axis.pop(0)
        dimension = dimension - 1
    temp = torch.flatten(temp, start_dim=dimension - len(axis))
    ret = torch.quantile(
        temp, qt, dim=-1, keepdim=keepdims, interpolation=interpolation, out=out
    )
    if keepdims:
        keepdim_shape = tuple(keepdim_shape)
        ret = ret.reshape(keepdim_shape)
    return ret


quantile.support_native_out = True


def corrcoef(
    x: torch.Tensor,
    /,
    *,
    y: Optional[torch.Tensor] = None,
    rowvar: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = torch.concat([x, y], dim=axis)
        xarr = xarr.T if not rowvar else xarr

    return torch.corrcoef(xarr)


def nanmedian(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def bincount(
    x: torch.Tensor,
    /,
    *,
    weights: Optional[torch.Tensor] = None,
    minlength: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        ret = torch.bincount(x, minlength=minlength)
        ret = ret.to(x.dtype)
    else:
        ret = torch.bincount(x, weights=weights, minlength=minlength)
        ret = ret.to(weights.dtype)
    return ret


bincount.support_native_out = False
