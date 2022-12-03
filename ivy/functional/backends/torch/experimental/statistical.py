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

@with_unsupported_dtypes({"1.11.0 and below":
                          ("bfloat16", "bfloat32", "float16")}, backend_version)
def quantile(
    a: torch.tensor,
    q: Union[torch.tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = 'linear',
    out: Optional[torch.tensor] = None
) -> torch.tensor:

    if axis is None:
        return torch.quantile(a,
                              q,
                              keepdim=keepdims,
                              interpolation=interpolation)

    if isinstance(axis, list) or isinstance(axis, tuple):
        for i in axis:
            a = torch.quantile(a,
                               q,
                               i,
                               keepdim=keepdims, 
                               interpolation=interpolation)
        return a

    return torch.quantile(a,
                          q, 
                          dim=axis,
                          keepdim=keepdims,
                          interpolation=interpolation)

