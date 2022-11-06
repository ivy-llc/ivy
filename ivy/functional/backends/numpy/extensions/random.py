# global
from typing import Optional, Union, Sequence
import numpy as np

# local
import ivy
from ivy.functional.ivy.random import _check_bounds_and_get_shape


# dirichlet
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


def beta(
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str = None,
    dtype: np.dtype = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape)
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(np.random.beta(alpha, beta, shape), dtype=dtype)


def gamma(
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str = None,
    dtype: np.dtype = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape)
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(np.random.gamma(alpha, beta, shape), dtype=dtype)
