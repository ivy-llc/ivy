import tensorflow as tf
from typing import Optional
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.13.0 and below": "bool"}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": "bool"}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("bool", "bfloat16")}, backend_version)
def kl_div(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> tf.Tensor:
    size = tf.shape(input)

    loss = tf.reduce_sum(input * tf.math.log(input / target), axis=-1)

    if reduction == "mean":
        loss = tf.math.reduce_mean(loss)
    elif reduction == "sum":
        loss = tf.math.reduce_sum(loss)
    elif reduction == "batchmean":
        loss = tf.math.reduce_sum(loss) / tf.cast(size[0], dtype=tf.float32)

    return loss
