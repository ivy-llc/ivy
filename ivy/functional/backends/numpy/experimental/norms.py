import numpy as np
from typing import Optional
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.25.2 and below": ("float16",)}, backend_version)
def l1_normalize(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        norm = np.sum(np.abs(x.flatten()))
        denorm = norm * np.ones_like(x)
    else:
        norm = np.sum(np.abs(x), axis=axis, keepdims=True)
        denorm = np.divide(norm, np.abs(x) + 1e-12)
    return np.divide(x, denorm)


def l2_normalize(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        denorm = np.linalg.norm(x.flatten(), 2, axis)
    else:
        denorm = np.linalg.norm(x, 2, axis, keepdims=True)
    denorm = np.maximum(denorm, 1e-12)
    return x / denorm


def lp_normalize(
    x: np.ndarray,
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        denorm = np.linalg.norm(x.flatten(), axis=axis, ord=p)
    else:
        denorm = np.linalg.norm(x, axis=axis, ord=p, keepdims=True)
    denorm = np.maximum(denorm, 1e-12)
    return np.divide(x, denorm, out=out)
