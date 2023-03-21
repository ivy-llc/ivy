from typing import Optional, Union

# global
import torch
import torch.nn

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def logit(
    x: torch.Tensor,
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.logit(x, eps=eps, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def thresholded_relu(
    x: torch.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.threshold(x, threshold=threshold, value=0)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def relu6(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.relu6(x)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def batch_norm(
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    /,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    training: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean.requires_grad = False
    variance.requires_grad = False
    scale.requires_grad = False
    offset.requires_grad = False
    return torch.nn.functional.batch_norm(
        x, mean, variance, weight=scale, bias=offset, training=training, eps=eps
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def log_sigmoid(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.logsigmoid(x)


def softsign(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.softsign(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def silu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.silu(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def selu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.selu(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def hard_sigmoid(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.hardsigmoid(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def hard_silu(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.multiply(x, hard_sigmoid(x))


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def hard_tanh(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.where(x > 1, 1, torch.where(x < -1, -1, x))


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def leaky_relu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def elu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.elu(x, alpha=alpha)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def celu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.celu(x, alpha=alpha)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def glu(
    x: torch.Tensor, /, *, axis: int = -1, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.nn.functional.glu(x, dim=axis)
