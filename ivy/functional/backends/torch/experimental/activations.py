# global
import torch
from typing import Optional

# local
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def logit(x: torch.Tensor, /, *, eps: Optional[float] = None, out=None):
    return torch.logit(x, eps=eps, out=out)
