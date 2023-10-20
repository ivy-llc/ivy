import jax.numpy as jnp
import jax
from typing import Optional
from ivy.functional.backends.jax import JaxArray
import ivy

# local
from ivy.func_wrapper import (
    with_supported_device_and_dtypes,
)
from . import backend_version


def huber_loss(
    input: JaxArray, target: JaxArray, /, *, delta: float = 1.0, reduction: str = "mean"
) -> JaxArray:
    residual = jnp.abs(input - target)
    quadratic_loss = 0.5 * (residual**2)
    linear_loss = delta * residual - 0.5 * (delta**2)
    loss = jnp.where(residual < delta, quadratic_loss, linear_loss)

    if reduction == "mean":
        loss = jnp.mean(loss)
    elif reduction == "sum":
        loss = jnp.sum(loss)

    return loss


def smooth_l1_loss(
    input: JaxArray,
    target: JaxArray,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> JaxArray:
    if beta < 1e-5:
        loss = jnp.abs(input - target)
    else:
        diff = jnp.abs(input - target)
        loss = jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def soft_margin_loss(
    input: JaxArray,
    target: JaxArray,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> JaxArray:
    loss = jnp.sum(jnp.log1p(jnp.exp(-input * target))) / jnp.size(input)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def _apply_loss_reduction(loss: JaxArray, reduction: str, axis=None) -> JaxArray:
    if reduction == "sum":
        return jnp.sum(loss, axis=axis)
    elif reduction == "mean":
        return jnp.mean(loss, axis=axis)
    else:  # reduction == "none"
        return loss


def _validate_poisson_nll_params(
    input,
    label,
    epsilon,
    reduction,
    allowed_dtypes=["float16", "float32", "float64"],
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
        "0.4.14 and below": {
            "cpu": ("float16", "float32", "float64"),
        }
    },
    backend_version,
)
def poisson_nll_loss(
    input: JaxArray,
    target: JaxArray,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> JaxArray:
    input_arr = jnp.asarray(input, dtype=input.dtype)
    target_arr = jnp.asarray(target, dtype=input.dtype)

    # check params
    _validate_poisson_nll_params(input_arr, target_arr, eps, reduction)

    if log_input:
        loss = jnp.exp(input_arr) - target_arr * input_arr
    else:
        loss = input_arr - target_arr * jnp.log(input_arr + eps)

    if full:
        point_five = jnp.array(0.5, dtype=target_arr.dtype)
        two_pi = jnp.array(2 * jnp.pi, dtype=target_arr.dtype)
        striling_approx_term = (
            (target_arr * jnp.log(target_arr))
            - target_arr
            + (point_five * jnp.log(two_pi * target_arr))
        )
        zeroes = jnp.zeros_like(target_arr, dtype=target_arr.dtype)
        ones = jnp.ones_like(target_arr, dtype=target_arr.dtype)
        cond = jnp.logical_and(target_arr >= zeroes, target_arr <= ones)
        loss = loss + jnp.where(cond, zeroes, striling_approx_term)
    return _apply_loss_reduction(loss, reduction)


@with_supported_device_and_dtypes(
    {
        "0.4.14 and below": {
            "cpu": ("float16", "float32", "float64"),
        }
    },
    backend_version,
)
def binary_cross_entropy(
    input: JaxArray,
    target: JaxArray,
    /,
    *,
    from_logits: bool = False,
    epsilon: float = 1e-7,
    reduction: str = "none",
    pos_weight: Optional[JaxArray] = None,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to jnp.binary_cross_entropy is not supported."
        )

    input_arr = jnp.asarray(input, dtype=input.dtype)
    target_arr = jnp.asarray(target, dtype=input.dtype)

    if from_logits:
        input = jax.nn.sigmoid(input_arr)
        if pos_weight is not None:
            pos_weight = jnp.asarray(pos_weight, dtype=input.dtype)
            num_classes = (
                input_arr.shape[0] if len(input_arr.shape) == 1 else input_arr.shape[1]
            )
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    "pos_weight must have the same size as the number of classes in"
                    " pred at non-singleton dimension 1"
                )
            loss = -1.0 * (
                (pos_weight * target_arr * jnp.log(input_arr + epsilon))
                + (1.0 - target_arr) * jnp.log(1.0 - input_arr + epsilon)
            )
        else:
            loss = -1.0 * (
                target_arr * jnp.log(input_arr + epsilon)
                + (1.0 - target_arr) * jnp.log(1.0 - input_arr + epsilon)
            )
    else:
        loss = -1.0 * (
            target_arr * jnp.log(input_arr + epsilon)
            + (1.0 - target_arr) * jnp.log(1.0 - input_arr + epsilon)
        )
    return _apply_loss_reduction(loss, reduction, axis=axis)

def _validate_nll_params(input, label, weight, reduction, allowed_dtypes=(jnp.float32, jnp.float64)):
    # Validate dtypes
    for parameter, name in zip([input, label], ["input", "label"]):
        if parameter.dtype not in allowed_dtypes:
            raise ValueError(
                f"The dtype of '{name}' in poisson_nll_loss should be one of {allowed_dtypes}, but"
                f" received {parameter.dtype}."
            )

    # Validate reduction
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            f"The value of 'reduction' in poisson_nll_loss should be 'sum', 'mean' or"
            f" 'none', but received {reduction}, which is not allowed."
        )

    # Validate shape
    if input.shape != label.shape:
        raise ValueError(
            f"The shape of 'input' ({input.shape}) must be the same as the shape of 'label' ({label.shape})."
        )

    return True

def _apply_loss_reduction(loss, reduction):
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
    else:  # reduction == "none"
        return loss

def nn_loss(input, target, weight=None, ignore_index=-100, reduction="mean"):
    _validate_nll_params(input, target, weight, reduction)

    flat_target = target.flatten()
    ignore_classes_mask = jnp.equal(flat_target, ignore_index)

    ignore_class_weight = jnp.array(0, dtype=input.dtype)

    if input.ndim == 1:
        current_weight = jnp.where(
            ignore_classes_mask,
            ignore_class_weight,
            weight[flat_target] if weight is not None else jnp.array(1, dtype=input.dtype)
        )
        loss = -input * current_weight
    elif input.ndim == 2:
        current_weight = jnp.where(
            ignore_classes_mask,
            ignore_class_weight,
            jnp.take(weight, target)
        )
        loss = -jnp.take(input, jnp.stack((jnp.arange(input.shape[0]), target), axis=-1)) * current_weight
    else:
        batch_size = input.shape[0]
        extent = input.shape[1]
        indices = jnp.arange(batch_size * extent)
        bdx = indices // extent
        kdx = indices % extent
        current_weight = jnp.where(
            ignore_classes_mask,
            ignore_class_weight,
            weight[flat_target] if weight is not None else jnp.array(1, dtype=input.dtype)
        )
        loss = -jnp.take(input, jnp.stack([bdx, flat_target, kdx], axis=-1)) * current_weight
        loss = jnp.reshape(loss, target.shape)

    if reduction == 'mean':
        return jnp.sum(loss) / jnp.sum(current_weight)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss
