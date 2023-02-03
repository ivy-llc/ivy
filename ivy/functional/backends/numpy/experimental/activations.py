from typing import Optional, Union

# global
import numpy as np

# local
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array


def logit(x: np.ndarray, /, *, eps: Optional[float] = None, out=None):
    x_dtype = x.dtype
    if eps is None:
        x = np.where(np.logical_or(x > 1, x < 0), np.nan, x)
    else:
        x = np.clip(x, eps, 1 - eps)
    ret = (np.log(x / (1 - x))).astype(x_dtype)
    if np.isscalar(ret):
        return np.array(ret)
    return ret


@_scalar_output_to_0d_array
def thresholded_relu(
    x: np.ndarray,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.where(x > threshold, x, 0).astype(x.dtype)


thresholded_relu.support_native_out = True
