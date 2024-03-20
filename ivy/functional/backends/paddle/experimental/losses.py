# global
from typing import Optional
import paddle
import paddle.nn.functional as F
import math

# local
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
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
        "2.6.0 and below": {
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
        input, target, reduction=reduction, delta=beta
    )


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
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
        "2.6.0 and below": {
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


@with_supported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float32", "float64")}},
    backend_version,
)
def kl_div(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
    log_target=False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if log_target:
        target = paddle.exp(target)
    loss = F.kl_div(input, target, reduction=reduction)
    return loss


def _apply_loss_reduction(loss: paddle.Tensor, reduction: str) -> paddle.Tensor:
    if reduction == "sum":
        return paddle.sum(loss)
    elif reduction == "mean":
        return paddle.mean(loss)
    else:  # reduction == "none"
        return loss


def _validate_poisson_nll_params(
    input,
    label,
    epsilon,
    reduction,
    allowed_dtypes=[paddle.float32, paddle.float64],
):
    # Validate dtypes
    for parameter, name in zip([input, label], ["input", "label"]):
        if parameter.dtype not in allowed_dtypes:
            raise ValueError(
                f"The dtype of '{name}' in poisson_nll_loss should be one of"
                f" {allowed_dtypes}, but received {parameter.dtype}."
            )

    # Validate epsilon
    if epsilon <= 0:
        raise ValueError(
            "The value of `epsilon` in poisson_nll_loss should be positive, but"
            f" received {epsilon}, which is not allowed."
        )

    # Validate reduction
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in poisson_nll_loss should be 'sum', 'mean' or"
            f" 'none', but received {reduction}, which is not allowed."
        )

    # Validate shape
    if input.shape != label.shape:
        raise ValueError(
            f"The shape of 'input' ({input.shape}) must be the same as the shape of"
            f" 'label' ({label.shape})."
        )

    return True


@with_supported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
# Note: This is a composition function to address an issue with the native
# `paddle.nn.functional.poisson_nll_loss` function. Once PaddlePaddle moves the
# changes from the develop branch to a stable release, this function can be replaced
# by the native implementation.
# Refer to the PR for more details: https://github.com/PaddlePaddle/Paddle/pull/56992
def poisson_nll_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> paddle.Tensor:
    input_arr = paddle.to_tensor(input)
    target_arr = paddle.to_tensor(target, dtype=input.dtype)

    _validate_poisson_nll_params(input_arr, target_arr, eps, reduction)

    if log_input:
        loss = paddle.exp(input_arr) - target_arr * input_arr
    else:
        loss = input_arr - target_arr * paddle.log(input_arr + eps)

    if full:
        point_five = paddle.to_tensor(0.5, dtype=target_arr.dtype)
        two_pi = paddle.to_tensor(2 * math.pi, dtype=target_arr.dtype)
        striling_approx_term = (
            (target_arr * paddle.log(target_arr))
            - target_arr
            + (point_five * paddle.log(two_pi * target_arr))
        )
        zeroes = paddle.zeros_like(target_arr, dtype=target_arr.dtype)
        ones = paddle.ones_like(target_arr, dtype=target_arr.dtype)
        cond = paddle.logical_and(target_arr >= zeroes, target_arr <= ones)
        loss = loss + paddle.where(cond, zeroes, striling_approx_term)
    return _apply_loss_reduction(loss, reduction)


@with_supported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("float16", "float32", "float64"),
        }
    },
    backend_version,
)
def hinge_embedding_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> paddle.Tensor:
    return paddle.nn.functional.hinge_embedding_loss(
        input,
        target,
        margin=margin,
        reduction=reduction,
    )
