# global
from typing import Optional, Union, Tuple
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
