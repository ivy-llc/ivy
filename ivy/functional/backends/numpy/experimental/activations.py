import numpy as np
from typing import Optional


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
