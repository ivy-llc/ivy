from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


@with_unsupported_dtypes(
    {"2.0.1 and below": ("unit8", "int8", "int16", "int32", "int64", "bool")},
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
        "2.0.1 and below": (
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
    {"2.0.1 and below": ("uint8", "int8", "int16", "int32", "int64", "bool")},
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
        "2.0.1 and below": (
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


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
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
def kl_div(
    input: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    loss = torch.nn.functional.kl_div(
        input,
        target,
        reduction=reduction,
    )
    return loss
