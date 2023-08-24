import tensorflow as tf
from typing import Optional
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.13.0 and below": "bool"}, backend_version)
def smooth_l1_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    delta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> tf.Tensor:
    abs_diff = ivy.abs(input - target)
    quadratic_loss = 0.5 * (abs_diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = ivy.where(abs_diff <= delta, quadratic_loss, linear_loss)

    if reduction == "sum":
        return ivy.sum(loss)
    elif reduction == "mean":
        return ivy.mean(loss)
    else:
        return loss


@with_unsupported_dtypes({"2.13.0 and below": "bool"}, backend_version)
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
