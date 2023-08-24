import numpy as np
from typing import Optional
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


# Implementation of smooth_l1_loss in the given format
@with_unsupported_dtypes({"1.25.2 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def huber_loss(
    input: np.ndarray,
    target: np.ndarray,
    /,
    *,
    delta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> np.ndarray:
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
