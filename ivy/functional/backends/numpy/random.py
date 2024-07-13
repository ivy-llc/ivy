"""Collection of Numpy random functions, wrapped to fit Ivy syntax and
signature."""

# global
import numpy as np
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, np.ndarray] = 0.0,
    high: Union[float, np.ndarray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int], np.ndarray]] = None,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if seed:
        np.random.seed(seed)
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    return np.asarray(np.random.uniform(low, high, shape), dtype=dtype)


def random_normal(
    *,
    mean: Union[float, np.ndarray] = 0.0,
    std: Union[float, np.ndarray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: np.dtype,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape
    if seed:
        np.random.seed(seed)
    return np.asarray(np.random.normal(mean, std, shape), dtype=dtype)


@with_unsupported_dtypes({"1.26.3 and below": ("bfloat16",)}, backend_version)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[np.ndarray] = None,
    replace: bool = True,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if seed:
        np.random.seed(seed)
    if probs is None:
        probs = (
            np.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = np.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / np.sum(probs_flat, -1, keepdims=True, dtype="float64")
    probs_stack = np.split(probs_flat, probs_flat.shape[0])
    samples_stack = [
        np.random.choice(num_classes, num_samples, replace, p=prob[0])
        for prob in probs_stack
    ]
    samples_flat = np.stack(samples_stack)
    return np.asarray(np.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


def randint(
    low: Union[float, np.ndarray],
    high: Union[float, np.ndarray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[np.dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    if seed:
        np.random.seed(seed)
    return np.random.randint(low, high, shape, dtype=dtype)


def seed(*, seed_value: int = 0):
    np.random.seed(seed_value)
    return


def shuffle(
    x: np.ndarray,
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if seed:
        np.random.seed(seed)
    if len(x.shape) == 0:
        return x

    x = np.array(x)
    rng = np.random.default_rng()
    rng.shuffle(x, axis=axis)

    return x
