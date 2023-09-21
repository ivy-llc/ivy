from typing import Optional

import numpy as np

from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array

from . import backend_version


@with_unsupported_dtypes({"1.26.0 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def huber_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    delta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
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
@with_unsupported_dtypes({"1.26.0 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def smooth_l1_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
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


@with_unsupported_dtypes({"1.26.0 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def soft_margin_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> np.ndarray:
    loss = np.sum(np.log1p(np.exp(-input * target))) / input.size

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss


@with_unsupported_dtypes({"1.26.0 and below": ("bool", "bfloat16")}, backend_version)
@_scalar_output_to_0d_array
def kl_div(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> np.ndarray:
    size = np.shape(input)

    loss = np.sum(input * np.log(input / target), axis=-1)

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    elif reduction == "batchmean":
        loss = np.divide(np.sum(loss), size[0])

    return loss
