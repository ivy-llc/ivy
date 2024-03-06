"""PyTorch activation functions.

Collection of PyTorch activation functions, wrapped to fit Ivy syntax
and signature.
"""

from typing import Optional, Union, Literal

# global
import numpy as np
import torch
import torch.nn

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
import ivy.functional.backends.torch as torch_backend


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bool",
        )
    },
    backend_version,
)
def relu(
    x: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.relu(x)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def leaky_relu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 0.2,
    complex_mode="jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def gelu(
    x: torch.Tensor,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if approximate:
        return 0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
    return torch.nn.functional.gelu(x)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def sigmoid(
    x: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not ivy.is_array(x):
        x = torch.tensor(x)
    return torch.sigmoid(x, out=out)


sigmoid.support_native_out = True


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    if torch.is_complex(x):
        amax = torch_backend.max(x, axis=axis, keepdims=True)
        exp_x = torch.exp(torch.subtract(x, amax))
        return torch.divide(exp_x, torch.sum(exp_x, dim=axis, keepdim=True))
    return torch.nn.functional.softmax(x, axis)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def softplus(
    x: torch.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode="jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = {
        k: v for k, v in {"beta": beta, "threshold": threshold}.items() if v is not None
    }
    return torch.nn.functional.softplus(x, **kwargs)


# Softsign
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def softsign(x: torch.Tensor, /, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # return x / (1 + torch.abs(x))
    return torch.nn.functional.softsign(x)


softsign.support_native_out = True


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    backend_version,
)
def log_softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[torch.Tensor] = None,
):
    if torch.is_complex(x):
        x_max = torch_backend.max(x, axis=axis, keepdims=True)
        sub_temp = torch.sub(x, x_max)
        ret = torch.sum(sub_temp.exp(), dim=axis, keepdim=True)
        ret = torch.log(ret)
        return torch.sub(sub_temp, ret)
    return torch.nn.functional.log_softmax(x, axis)


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    backend_version,
)
def mish(
    x: torch.Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if torch.is_complex(x):
        x_norm = torch.log1p(x.exp())
        return torch.multiply(x, x_norm.tanh())
    return torch.nn.functional.mish(x)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "complex",
            "float16",
        )
    },
    backend_version,
)
def hardswish(
    x: torch.Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.hardswish(x)
