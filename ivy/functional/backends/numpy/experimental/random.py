# global
from typing import Optional, Union, Sequence
import numpy as np

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)


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
    device: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(np.random.beta(alpha, beta, shape), dtype=dtype)


def gamma(
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(np.random.gamma(alpha, beta, shape), dtype=dtype)


def poisson(
    lam: Union[float, np.ndarray],
    *,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    lam = np.array(lam)

    if seed:
        np.random.seed(seed)
    if shape is not None:
        _check_shapes_broadcastable(lam.shape, shape)
    if np.any(lam < 0):
        pos_lam = np.where(lam < 0, 0, lam)
        ret = np.random.poisson(pos_lam, shape)
        ret = np.where(lam < 0, fill_value, ret)
    else:
        ret = np.random.poisson(lam, shape)
    return np.asarray(ret, dtype=dtype)


def bernoulli(
    probs: Union[float, np.ndarray],
    *,
    logits: Optional[Union[float, np.ndarray]] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if logits is not None:
        probs = np.asarray(ivy.softmax(logits), dtype=dtype)
    if not _check_shapes_broadcastable(shape, probs.shape):
        shape = probs.shape
    return np.asarray(np.random.binomial(1, p=probs, size=shape), dtype=dtype)
