"""
Collection of PyTorch activation functions, wrapped to fit Ivy syntax and signature.
"""

from typing import Optional

# global
import numpy as np
import torch

# local
import ivy


def relu(x: torch.Tensor,
         out: Optional[torch.Tensor] = None)\
        -> torch.Tensor:
    ret = torch.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: torch.Tensor, alpha: Optional[float] = 0.2)\
        -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


def gelu(x, approximate: bool = True):
    if approximate:
        return 0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x ** 3)))
    return torch.nn.functional.gelu(x)


def tanh(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.tanh(x)



def sigmoid(x):
    return torch.sigmoid(x)


def softmax(x: torch.Tensor, axis: Optional[int] = -1)\
        -> torch.Tensor:
    return torch.softmax(x, axis)


def softplus(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.nn.functional.softplus(x)
