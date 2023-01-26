import numpy as np
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def l2_normalize(x: np.ndarray, /, *, axis: int = None, out=None) -> np.ndarray:
    denorm = np.linalg.norm(x, axis=axis, ord=2, keepdims=True)
    denorm = np.maximum(denorm, 1e-12)
    return x / denorm
