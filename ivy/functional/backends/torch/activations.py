"""Collection of PyTorch activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import numpy as np
import torch
import torch.nn
# local
from torch.overrides import handle_torch_function, has_torch_function_unary

import ivy


def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: torch.Tensor, alpha: Optional[float] = 0.2)-> torch.Tensor:
    return torch.nn.functional.leaky_relu(x,alpha)

def gelu(x: torch.Tensor)->torch.Tensor:
    return torch.nn.functional.gelu(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def softmax(x: torch.Tensor, axis: Optional[int] = -1) -> torch.Tensor:
    return torch.nn.functional.softmax(x,axis)


def softplus(x: torch.Tensor,approximate = False) -> torch.Tensor:
    return torch.nn.functional.softplus(x,approximate)
