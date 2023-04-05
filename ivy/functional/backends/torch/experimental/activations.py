from typing import Optional, Union

# global
import torch
import torch.nn

# local
import ivy
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
def hardshrink(
    x: torch.Tensor,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[torch.Tensor] = None,
):
    return torch.nn.functional.hardshrink(x, lambd=lambd)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def softshrink(
    x: torch.Tensor,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[torch.Tensor] = None,
):
    return torch.nn.functional.softshrink(x, lambd=lambd)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def thresholded_relu(
    x: torch.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x, threshold = ivy.promote_types_of_inputs(x, threshold)
    return torch.threshold(x, threshold=threshold, value=0)


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "float16")}, backend_version)
def threshold(
    x: torch.Tensor,
    threshold: Optional[Union[int, float]],
    value: Optional[Union[int, float]],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.threshold(x, threshold, value)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def relu6(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nn.functional.relu6(x)


relu6.unsupported_dtypes = (
    "float16",
    "bfloat16",
)


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
):
    mean.requires_grad = False
    variance.requires_grad = False
    scale.requires_grad = False
    offset.requires_grad = False
    return torch.nn.functional.batch_norm(
        x, mean, variance, weight=scale, bias=offset, training=training, eps=eps
    )


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def group_norm(
    x: torch.Tensor,
    num_groups: int,
    /,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return torch.nn.functional.group_norm(
        x, num_groups, weight=weight, bias=bias, eps=eps
    )


@with_unsupported_dtypes({"1.13.0 and below": ("float16", "bfloat16")}, backend_version)
def logsigmoid(
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
    return x * torch.nn.functional.hardsigmoid(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def hard_tanh(
    x: torch.Tensor,
    /,
    *,
    min_value: float = -1.0,
    max_value: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.hardtanh(x, min_val=min_value, max_val=max_value)


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
def parametric_relu(
    x: torch.Tensor,
    weight: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.prelu(x, weight)


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
