# global
from typing import Optional, Union, Tuple
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _consistent_median(input, axis, keepdims, out):
    input_max = torch.max(input, axis, keepdim=True)[0]
    return torch.div(
        torch.add(
            torch.median(
                torch.cat((input, input_max), dim=axis),
                dim=axis,
                keepdim=keepdims,
            )[0],
            torch.median(
                input,
                dim=axis,
                keepdim=keepdims,
            )[0],
        ),
        2,
        out=out,
    )


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
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
            input = _consistent_median(input, dim, keepdims, out)
    elif axis is None:
        input = _consistent_median(torch.flatten(input), -1, keepdims, out)
    else:
        input = _consistent_median(input, axis, keepdims, out)
    return input


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


def unravel_index(
    indices: torch.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = []
    for dim in reversed(shape):
        output.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(output))
