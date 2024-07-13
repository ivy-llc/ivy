import numpy as np
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_dtypes({"1.26.3 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def huber_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    delta: float = 1.0,
    reduction: str = "mean",
) -> np.ndarray:
    abs_diff = np.abs(input - target)
    quadratic_loss = 0.5 * (abs_diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = np.where(abs_diff <= delta, quadratic_loss, linear_loss)

    if reduction == "sum":
        return np.sum(loss)
    elif reduction == "mean":
        return np.mean(loss)
    else:
        return loss


# Implementation of smooth_l1_loss in the given format
@with_unsupported_dtypes({"1.26.3 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def smooth_l1_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
) -> np.ndarray:
    if beta < 1e-5:
        loss = np.abs(input - target)
    else:
        diff = np.abs(input - target)
        loss = np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss


@with_unsupported_dtypes({"1.26.3 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def soft_margin_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    reduction: str = "mean",
) -> np.ndarray:
    loss = np.sum(np.log1p(np.exp(-input * target))) / input.size

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss


def _apply_loss_reduction(loss: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "sum":
        return np.sum(loss)
    elif reduction == "mean":
        return np.mean(loss)
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
        "1.26.0 and below": {
            "cpu": ("float16", "float32", "float64"),
        }
    },
    backend_version,
)
@_scalar_output_to_0d_array
def poisson_nll_loss(
    input: np.ndarray,
    target: np.ndarray,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> np.ndarray:
    input_arr = np.asarray(input)
    target_arr = np.asarray(target, dtype=input.dtype)

    _validate_poisson_nll_params(input_arr, target_arr, eps, reduction)

    if log_input:
        loss = np.exp(input_arr) - target_arr * input_arr
    else:
        loss = input_arr - target_arr * np.log(input_arr + eps)

    if full:
        point_five = np.array(0.5, dtype=target_arr.dtype)
        two_pi = np.array(2 * np.pi, dtype=target_arr.dtype)
        striling_approx_term = (
            (target_arr * np.log(target_arr))
            - target_arr
            + (point_five * np.log(two_pi * target_arr))
        )
        zeroes = np.zeros_like(target_arr, dtype=target_arr.dtype)
        ones = np.ones_like(target_arr, dtype=target_arr.dtype)
        cond = np.logical_and(target_arr >= zeroes, target_arr <= ones)
        loss = loss + np.where(cond, zeroes, striling_approx_term)
    return _apply_loss_reduction(loss, reduction)


@with_supported_device_and_dtypes(
    {
        "1.26.0 and below": {
            "cpu": ("float32", "float64"),
        }
    },
    backend_version,
)
def hinge_embedding_loss(
    input: np.ndarray,
    target: np.ndarray,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> np.ndarray:
    zero_ = np.zeros([1], dtype=input.dtype)

    relu_part = np.maximum(margin - input, 0)

    loss = np.where(target == 1.0, input, zero_) + np.where(
        target == -1.0, relu_part, zero_
    )

    return _apply_loss_reduction(loss, reduction)
