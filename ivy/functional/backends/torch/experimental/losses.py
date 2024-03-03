from typing import Optional
import torch
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
    with_supported_dtypes,
)
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes(
    {"2.2 and below": ("unit8", "int8", "int16", "int32", "int64", "bool")},
    backend_version,
)
def l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    return torch.nn.functional.l1_loss(
        input,
        target,
        reduction=reduction,
    )


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "complex",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    backend_version,
)
def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(
        input,
        target,
        beta=beta,
        reduction=reduction,
    )


@with_unsupported_dtypes(
    {"2.2 and below": ("uint8", "int8", "int16", "int32", "int64", "bool")},
    backend_version,
)
def huber_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
    delta: Optional[float] = 1.0,
) -> torch.Tensor:
    return torch.nn.functional.huber_loss(
        input, target, reduction=reduction, delta=delta
    )


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    backend_version,
)
def soft_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    return torch.nn.functional.soft_margin_loss(
        input,
        target,
        reduction=reduction,
    )


@with_supported_dtypes(
    {"2.2 and below": ("float",)},
    backend_version,
)
def kl_div(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
    log_target=False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = torch.nn.functional.kl_div(
        input, target, reduction=reduction, log_target=log_target
    )
    return loss


@with_supported_device_and_dtypes(
    {
        "2.15.0 and below": {
            "cpu": (
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "complex64",
                "complex128",
            ),
        }
    },
    backend_version,
)
def poisson_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    return torch.nn.functional.poisson_nll_loss(
        input, target, log_input=log_input, full=full, eps=eps, reduction=reduction
    )


@with_supported_device_and_dtypes(
    {
        "2.2 and below": {
            "cpu": ("float16", "float32", "float64"),
            "gpu": ("float16", "float32", "float64"),
        }
    },
    backend_version,
)
def hinge_embedding_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return torch.nn.functional.hinge_embedding_loss(
        input, target, margin=margin, reduction=reduction
    )
