# global
import ivy
from typing import Optional, Union, Tuple
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def histogram(
    a: torch.tensor,
    /,
    *,
    bins: Optional[Union[int, torch.tensor, str]] = None,
    axis: Optional[torch.Tensor] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[torch.tensor] = None,
    density: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> Tuple[torch.tensor]:

    ret = torch.histogram(
        input=a, bins=bins, range=range, weight=weights, density=density, out=out
    )

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    else:
        return ret


histogram.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def median(
    input: torch.tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    if hasattr(axis, "__iter__"):
        for dim in axis:
            input = torch.median(
                input,
                dim=dim,
                keepdim=keepdims,
                out=out,
            )[0]
        return input
    else:
        return torch.median(
            input,
            dim=axis,
            keepdim=keepdims,
            out=out,
        )


def nanmean(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmean(a, axis=axis, keepdim=keepdims, dtype=dtype, out=out)


nanmean_support_native_out = True
