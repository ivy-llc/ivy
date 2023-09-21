# global
from typing import Optional
import paddle
import paddle.nn.functional as F

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
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
def huber_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    delta: Optional[float] = 1.0,
) -> paddle.Tensor:
    return paddle.fluid.layers.huber_loss(input, target, delta=delta)


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
def soft_margin_loss(
    input: paddle.Tensor,
    label: paddle.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> paddle.Tensor:
    return paddle.nn.functional.soft_margin_loss(input, label, reduction=reduction)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "bfloat16",
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
def kl_div(
    input: paddle.Tensor, target: paddle.Tensor, /, *, reduction: Optional[str] = "mean"
) -> paddle.Tensor:
    loss = F.kl_div(input, target, reduction=reduction)
    return loss
