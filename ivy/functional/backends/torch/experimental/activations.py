from typing import Optional, Union

# global
import torch
import torch.nn

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def logit(x: torch.Tensor, /, *, eps: Optional[float] = None, out=None):
    return torch.logit(x, eps=eps, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def thresholded_relu(
    x: torch.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.threshold(x, threshold=threshold, value=0)
