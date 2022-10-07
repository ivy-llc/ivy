"""Collection of CUPY random functions, wrapped to fit Ivy syntax and signature."""

# global
import cupy as cp
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)

# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, cp.ndarray] = 0.0,
    high: Union[float, cp.ndarray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    shape = _check_bounds_and_get_shape(low, high, shape)
    return cp.asarray(cp.random.uniform(low, high, shape), dtype=dtype)


def random_normal(
    *,
    mean: Union[float, cp.ndarray] = 0.0,
    std: Union[float, cp.ndarray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
    dtype: cp.dtype,
    seed: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    if seed is not None:
        cp.random.seed(seed)
    return cp.asarray(cp.random.normal(mean, std, shape), dtype=dtype)


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[cp.ndarray] = None,
    replace: bool = True,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if probs is None:
        probs = (
            cp.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = cp.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / cp.sum(probs_flat, -1, keepdims=True, dtype="float64")
    probs_stack = cp.split(probs_flat, probs_flat.shape[0])
    samples_stack = [
        cp.random.choice(num_classes, num_samples, replace, p=prob[0])
        for prob in probs_stack
    ]
    samples_flat = cp.stack(samples_stack, out=out)
    return cp.asarray(cp.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


multinomial.support_native_out = True


def randint(
    low: Union[float, cp.ndarray],
    high: Union[float, cp.ndarray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
    dtype: Optional[Union[cp.dtype, ivy.Dtype]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape)
    return cp.random.randint(low, high, shape, dtype=dtype)


def seed(*, seed_value: int = 0) -> None:
    cp.random.seed(seed_value)


def shuffle(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.random.permutation(x)
