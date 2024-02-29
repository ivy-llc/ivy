from typing import Optional, Union, Literal

# global
import torch
import torch.nn

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def logit(
    x: torch.Tensor,
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.logit(x, eps=eps, out=out)


@with_unsupported_dtypes({"2.2 and below": ("complex", "float16")}, backend_version)
def thresholded_relu(
    x: torch.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.threshold(x, threshold=threshold, value=0)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def relu6(
    x: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.relu6(x)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def logsigmoid(
    input: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if torch.is_complex(input):
        return torch.log(torch.sigmoid(input))
    return torch.nn.functional.logsigmoid(input)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def selu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.nn.functional.selu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def silu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.silu(x)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def elu(
    x: torch.Tensor, /, *, alpha: float = 1.0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.nn.functional.elu(x, alpha)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "complex",
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def celu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 1.0,
    complex_mode="jax",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.celu(x, alpha=alpha)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def hardtanh(
    x: torch.Tensor,
    /,
    *,
    max_val: float = 1.0,
    min_val: float = -1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.nn.functional.hardtanh(x, max_val=max_val, min_val=min_val)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def tanhshrink(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.nn.functional.tanhshrink(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def threshold(
    x: torch.Tensor,
    /,
    *,
    threshold: float,
    value: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.nn.functional.threshold(threshold=threshold, value=value, input=x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def softshrink(
    x: torch.Tensor, /, *, lambd: float = 0.5, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.nn.functional.softshrink(x, lambd=lambd)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


def scaled_tanh(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return alpha * torch.nn.functional.tanh(beta * x)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def hardshrink(
    x: torch.Tensor, /, *, lambd: float = 0.5, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.nn.functional.hardshrink(x, lambd=lambd)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.0.1 and below": ("complex", "float16")}, backend_version)
def hardsilu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.nn.functional.hardswish(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)
