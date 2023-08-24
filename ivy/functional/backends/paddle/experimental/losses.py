# global
from typing import Optional
import paddle
import paddle.nn.functional as F

# local
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "float16",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> paddle.Tensor:
    return F.l1_loss(input, target, reduction=reduction)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def smooth_l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> paddle.Tensor:
    return paddle.nn.functional.smooth_l1_loss(
        input, target, reduction=reduction, beta=beta
    )


@with_supported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
def poisson_nll_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> paddle.Tensor:
    return paddle.nn.functional.poisson_nll_loss(
        input,
        target,
        log_input=log_input,
        full=full,
        epsilon=eps,
        reduction=reduction,
    )
