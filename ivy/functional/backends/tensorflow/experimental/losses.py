import tensorflow as tf
import math
from typing import Optional
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_dtypes({"2.15.0 and below": "bool"}, backend_version)
def huber_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    delta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> tf.Tensor:
    abs_diff = tf.abs(input - target)
    quadratic_loss = 0.5 * (abs_diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = tf.where(abs_diff <= delta, quadratic_loss, linear_loss)

    if reduction == "sum":
        return tf.sum(loss)
    elif reduction == "mean":
        return tf.mean(loss)
    else:
        return loss


@with_unsupported_dtypes({"2.15.0 and below": "bool"}, backend_version)
def smooth_l1_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> tf.Tensor:
    diff = tf.abs(input - target)
    loss = tf.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    if reduction == "mean":
        return tf.reduce_mean(loss)
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss


@with_unsupported_dtypes({"2.15.0 and below": "bool"}, backend_version)
def soft_margin_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> tf.Tensor:
    loss = tf.reduce_sum(tf.math.log1p(tf.exp(-input * target))) / tf.size(input)

    if reduction == "sum":
        return tf.reduce_sum(loss)
    elif reduction == "mean":
        return tf.reduce_mean(loss)
    else:
        return loss


def _apply_loss_reduction(loss: tf.Tensor, reduction: str) -> tf.Tensor:
    if reduction == "sum":
        return tf.math.reduce_sum(loss)
    elif reduction == "mean":
        return tf.reduce_mean(loss)
    else:  # reduction == "none"
        return loss


def _validate_poisson_nll_params(
    input,
    label,
    epsilon,
    reduction,
    allowed_dtypes=[tf.float32, tf.float64],
):
    # Validate dtypes
    for parameter, name in zip([input, label], ["input", "label"]):
        if parameter.dtype not in allowed_dtypes:
            raise TypeError(
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
        "2.15.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def poisson_nll_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> tf.Tensor:
    input_tensor = tf.constant(input, dtype=input.dtype)
    target_tensor = tf.constant(target, dtype=input.dtype)

    _validate_poisson_nll_params(input_tensor, target_tensor, eps, reduction)
    if log_input:
        loss = tf.math.exp(input_tensor) - target_tensor * input_tensor
    else:
        loss = input_tensor - target_tensor * tf.math.log(input_tensor + eps)
    if full:
        point_five = tf.constant(0.5, dtype=target_tensor.dtype)
        two_pi = tf.constant(2 * math.pi, dtype=target_tensor.dtype)

        stirling_approx = (
            (target_tensor * tf.math.log(target_tensor))
            - target_tensor
            + (point_five * tf.math.log(two_pi * target_tensor))
        )
        zeros = tf.zeros_like(target_tensor, dtype=target_tensor.dtype)
        ones = tf.ones_like(target_tensor, dtype=target_tensor.dtype)
        cond = tf.math.logical_and(target_tensor >= zeros, target_tensor <= ones)
        loss = loss + tf.where(cond, zeros, stirling_approx)
    return _apply_loss_reduction(loss, reduction)


@with_supported_device_and_dtypes(
    {
        "2.14.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def hinge_embedding_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> tf.Tensor:
    zero_ = tf.zeros([1], dtype=input.dtype)

    relu_part = tf.math.maximum(margin - input, 0)

    loss = tf.where(tf.equal(target, 1.0), input, zero_) + tf.where(
        tf.equal(target, -1.0), relu_part, zero_
    )

    return _apply_loss_reduction(loss, reduction)
