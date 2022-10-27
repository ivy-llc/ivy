from typing import Optional, Union, Sequence
import numpy as np
import ivy
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


# dirichlet
@with_supported_dtypes({"0.3.14 and below": ("float32", "float64")}, backend_version)
def dirichlet(
    alpha: Union[np.ndarray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    size = size if size is not None else len(alpha)
    dtype = dtype if dtype is not None else np.float64
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(np.random.dirichlet(alpha, size=size), dtype=dtype)


dirichlet.support_native_out = False
