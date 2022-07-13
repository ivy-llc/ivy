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


def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(
    x: torch.Tensor,
    alpha: Optional[float] = 0.2,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.nn.functional.leaky_relu(x, alpha)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def gelu(
    x : torch.Tensor,
    approximate: bool = True,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if approximate:
        ret = (0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x ** 3)
        )))
    else:
        ret = torch.nn.functional.gelu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tanh(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tanh(x, out=out)


def sigmoid(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sigmoid(x, out=out)


def softmax(
        x: torch.Tensor, axis: Optional[int] = None, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    exp_x = torch.exp(x)
    ret = exp_x / torch.sum(exp_x, axis, keepdims=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def softplus(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.softplus(x, out=out)
