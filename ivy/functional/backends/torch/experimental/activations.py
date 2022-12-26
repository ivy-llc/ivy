import torch
from typing import Optional


def logit(x: torch.Tensor, /, *, eps: Optional[float]=None, out=None):
    return torch.logit(x, eps=eps, out=out)
