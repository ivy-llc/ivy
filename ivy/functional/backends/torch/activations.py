"""
PyTorch activation functions.

Collection of PyTorch activation functions, wrapped to fit Ivy syntax
and signature.
"""
from typing import Optional, Union

# global
import numpy as np
import torch
import torch.nn

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def relu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.relu(x)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def leaky_relu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def gelu(
    x: torch.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if approximate:
        return (
            0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return torch.nn.functional.gelu(x)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def sigmoid(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not ivy.is_array(x):
        x = torch.tensor(x)
    return torch.sigmoid(x, out=out)


sigmoid.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex", "float16")}, backend_version)
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


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex", "float16", "bfloat16")}, backend_version
)
def softplus(
    x: torch.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = {
        k: v for k, v in {"beta": beta, "threshold": threshold}.items() if v is not None
    }
    return torch.nn.functional.softplus(x, **kwargs)


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "complex",
            "float16",
        )
    },
    backend_version,
)
def log_softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
):
    if axis is None:
        axis = -1
    return torch.nn.functional.log_softmax(x, axis)


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "complex",
            "float16",
        )
    },
    backend_version,
)
def mish(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.mish(x)


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "complex",
            "float16",
        )
    },
    backend_version,
)
def hardswish(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.hardswish(x)
