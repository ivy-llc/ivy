"""Collection of PyTorch activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional

# global
import numpy as np
import torch
import torch.nn

# local


import ivy


def relu(
    x: torch.Tensor, 
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


relu.support_native_out = True


def leaky_relu(
    x: torch.Tensor, 
    alpha: Optional[float] = 0.2,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


def gelu(
    x, 
    approximate: bool = True,
    *,
    out: Optional[torch.Tensor] = None
):
    if approximate:
        return (
            0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return torch.nn.functional.gelu(x)


def tanh(
    x: torch.Tensor, 
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.tanh(x, out=out)


tanh.support_native_out = True


def sigmoid(
    x: torch.Tensor, 
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.sigmoid(x, out=out)


sigmoid.support_native_out = True


def softmax(
    x: torch.Tensor, 
    axis: Optional[int] = None, 
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    exp_x = torch.exp(x, out=out)
    return exp_x / torch.sum(exp_x, axis, keepdims=True)


softmax.support_native_out = True


def softplus(
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.softplus(x)
