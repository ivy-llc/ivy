import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray

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
    beta: float = 1.0,
    reduction: str = "mean",
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
    reduction: str = "mean",
) -> JaxArray:
    loss = jnp.sum(jnp.log1p(jnp.exp(-input * target))) / jnp.size(input)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def _apply_loss_reduction(loss: JaxArray, reduction: str) -> JaxArray:
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
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
        "0.4.18 and below": {
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
        "0.4.18 and below": {
            "cpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def hinge_embedding_loss(
    input: JaxArray,
    target: JaxArray,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> JaxArray:
    zero_ = jnp.zeros([1], dtype=input.dtype)

    relu_part = jnp.maximum(margin - input, 0)

    loss = jnp.where(target == 1.0, input, zero_) + jnp.where(
        target == -1.0, relu_part, zero_
    )
    return _apply_loss_reduction(loss, reduction)
