import tensorflow as tf
import math
from typing import Optional
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_dtypes({"2.14.0 and below": "bool"}, backend_version)
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


@with_unsupported_dtypes({"2.14.0 and below": "bool"}, backend_version)
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


@with_unsupported_dtypes({"2.14.0 and below": "bool"}, backend_version)
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


@with_supported_device_and_dtypes(
    {
        "2.13.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def _validate_nll_params(
    input,
    label,
    weight,
    reduction,
    allowed_dtypes=[tf.float32, tf.float64],
):
    # Validate dtypes
    for parameter, name in zip([input, label], ["input", "label"]):
        if parameter.dtype not in allowed_dtypes:
            raise ValueError(
                "The dtype of '%s' in poisson_nll_loss should be one of %s, but"
                " received %s." % (name, allowed_dtypes, parameter.dtype)
            )

    # Validate reduction
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in poisson_nll_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction
        )

    # Validate shape
    if input.shape != label.shape:
        raise ValueError(
            "The shape of 'input' (%s) must be the same as the shape of 'label' (%s)."
            % (input.shape, label.shape)
        )

    return True


def nn_loss(
    input: tf.Tensor,
    target: tf.Tensor,
    *,
    weight: Optional[tf.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    _validate_nll_params(input, target, weight, reduction)

    flat_target = tf.reshape(target, [-1])
    ignore_classes_mask = tf.equal(flat_target, ignore_index)

    ignore_class_weight = tf.constant(0, dtype=input.dtype)

    if tf.rank(input) == 1:
        current_weight = tf.where(
            ignore_classes_mask,
            ignore_class_weight,
            (
                weight[flat_target]
                if weight is not None
                else tf.constant(1, dtype=input.dtype)
            ),
        )
        loss = -input * current_weight
    elif tf.rank(input) == 2:
        current_weight = tf.where(
            ignore_classes_mask, ignore_class_weight, tf.gather(weight, target)
        )
        loss = (
            -tf.gather_nd(
                input, tf.stack((tf.range(tf.shape(input)[0]), target), axis=-1)
            )
            * current_weight
        )
    else:
        print(input)
        batch_size = tf.shape(input)[0]
        extent = tf.shape(input)[1]
        indices = tf.range(batch_size * extent)
        bdx = indices // extent
        kdx = indices % extent
        current_weight = tf.where(
            ignore_classes_mask,
            ignore_class_weight,
            (
                weight[flat_target]
                if weight is not None
                else tf.constant(1, dtype=input.dtype)
            ),
        )
        loss = (
            -tf.gather_nd(input, tf.stack([bdx, flat_target, kdx], axis=-1))
            * current_weight
        )
        loss = tf.reshape(loss, tf.shape(target))

    if reduction == "mean":
        return tf.reduce_sum(loss) / tf.reduce_sum(current_weight)
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss


def _apply_loss_reduction(loss: tf.Tensor, reduction: str, axis) -> tf.Tensor:
    if reduction == "sum":
        return tf.math.reduce_sum(loss, axis=axis)
    elif reduction == "mean":
        return tf.reduce_mean(loss, axis=axis)
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
            raise ValueError(
                "The dtype of '%s' in poisson_nll_loss should be one of %s, but"
                " received %s." % (name, allowed_dtypes, parameter.dtype)
            )

    # Validate epsilon
    if epsilon <= 0:
        raise ValueError(
            "The value of `epsilon` in poisson_nll_loss should be positive, but"
            " received %f, which is not allowed" % epsilon
        )

    # Validate reduction
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in poisson_nll_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction
        )

    # Validate shape
    if input.shape != label.shape:
        raise ValueError(
            "The shape of 'input' (%s) must be the same as the shape of 'label' (%s)."
            % (input.shape, label.shape)
        )

    return True


@with_supported_device_and_dtypes(
    {
        "2.14.0 and below": {
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
        "2.13.0 and below": {
            "cpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def binary_cross_entropy(
    input: tf.Tensor,
    target: tf.Tensor,
    /,
    *,
    from_logits: bool = False,
    epsilon: float = 1e-7,
    reduction: str = "none",
    pos_weight: Optional[tf.Tensor] = None,
    axis: Optional[tf.Tensor] = None,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to tf.binary_cross_entropy is not supported."
        )

    input_tensor = tf.constant(input, dtype=input.dtype)
    target_tensor = tf.constant(target, dtype=input.dtype)

    if from_logits:
        input = tf.math.sigmoid(input_tensor)
        if pos_weight is not None:
            pos_weight = tf.constant(pos_weight, dtype=input.dtype)
            num_classes = (
                input_tensor.shape[0]
                if len(input_tensor.shape) == 1
                else input_tensor.shape[1]
            )
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    "pos_weight must have the same size as the number of classes in"
                    " pred at non-singleton dimension 1"
                )
            loss = -1.0 * (
                (pos_weight * target_tensor * tf.math.log(input_tensor + epsilon))
                + (1.0 - target_tensor) * tf.math.log(1.0 - input_tensor + epsilon)
            )
        else:
            loss = -1.0 * (
                target_tensor * tf.math.log(input_tensor + epsilon)
                + (1.0 - target_tensor) * tf.math.log(1.0 - input_tensor + epsilon)
            )
    else:
        loss = -1.0 * (
            target_tensor * tf.math.log(input_tensor + epsilon)
            + (1.0 - target_tensor) * tf.math.log(1.0 - input_tensor + epsilon)
        )
    return _apply_loss_reduction(loss, reduction, axis)
