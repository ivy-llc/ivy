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
from ivy.func_wrapper import with_unsupported_dtypes
from . import torch_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, torch_version)
def relu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.relu(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, torch_version)
def leaky_relu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, torch_version)
def gelu(
    x: torch.Tensor, /, *, approximate: bool = True, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if approximate:
        return (
            0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x ** 3)))
        )
    return torch.nn.functional.gelu(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, torch_version)
def sigmoid(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not ivy.is_array(x):
        x = torch.tensor(x)
    return torch.sigmoid(x, out=out)


sigmoid.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, torch_version)
def softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    return torch.nn.functional.softmax(x, axis)


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, torch_version)
def softplus(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.softplus(x)
