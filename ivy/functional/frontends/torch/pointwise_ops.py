# global
from typing import Optional

import torch

import ivy


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


tan.unsupported_dtypes = ('float16',)


def tanh(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return ivy.tanh(input, out=out)
