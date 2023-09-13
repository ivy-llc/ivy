import numpy as np
from typing import Optional
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


# Implementation of smooth_l1_loss in the given format
@with_unsupported_dtypes({"1.25.2 and below": ("bool",)}, backend_version)
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
