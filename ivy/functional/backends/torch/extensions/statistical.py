from typing import Optional, Union, Tuple
import torch


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
            )
        return input
    else:
        return torch.median(
            input,
            dim=axis,
            keepdim=keepdims,
            out=out,
        )
