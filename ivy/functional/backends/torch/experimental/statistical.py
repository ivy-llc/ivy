# global
from typing import Optional, Union, Tuple, Sequence
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def median(
    input: torch.tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return quantile(
        input,
        0.5,
        axis=axis,
        keepdims=keepdims,
        interpolation="midpoint",
    ).type_as(input)


median.support_native_out = False


def nanmean(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmean(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_unsupported_dtypes(
    {"1.11.0 and below": ("bfloat16", "bfloat32", "float16")}, backend_version
)
def quantile(
    a: torch.tensor,
    q: Union[torch.tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    temp = a.to(torch.float64)
    if isinstance(axis, list) or isinstance(axis, tuple):
        dimension = len(a.size())
        for x in axis:
            axis1 = x
            for axis2 in range(x + 1, dimension):
                temp = torch.transpose(temp, axis1, axis2)
                axis1 = axis2
        temp = torch.flatten(temp, start_dim=dimension - len(axis))
        return torch.quantile(
            temp, q, dim=-1, keepdim=keepdims, interpolation=interpolation, out=out
        )
    return torch.quantile(
        temp, q, dim=axis, keepdim=keepdims, interpolation=interpolation, out=out
    )


quantile.support_native_out = True


def corrcoef(
    x: torch.Tensor,
    /,
    *,
    y: Optional[torch.Tensor] = None,
    rowvar: Optional[bool] = True,
    out: Optional[torch.tensor] = None,
) -> torch.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = torch.concat([x, y], dim=axis)
        xarr = xarr.T if not rowvar else xarr

    return torch.corrcoef(xarr)


def nanmedian(
    input: torch.tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def unravel_index(
    indices: torch.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> Tuple:
    temp = indices.to(torch.int32)
    output = []
    for dim in reversed(shape):
        output.append(temp % dim)
        temp = temp // dim
    return tuple(reversed(output))


unravel_index.support_native_out = False


def bincount(
    x: torch.Tensor,
    /,
    *,
    weights: Optional[torch.Tensor] = None,
    minlength: Optional[int] = 0,
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
