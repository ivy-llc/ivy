"""Collection of MXNet random functions, wrapped to fit Ivy syntax and signature."""

# global
import mxnet as mx
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.functional.backends.mxnet import _1_dim_array_to_flat_array


# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, mx.nd.NDArray] = 0.0,
    high: Union[float, mx.nd.NDArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: mx.context.Context,
    dtype: type,
) -> mx.nd.NDArray:
    shape = _check_bounds_and_get_shape(low, high, shape)
    if isinstance(low, mx.nd.NDArray):
        low = low.asscalar()
    if isinstance(high, mx.nd.NDArray):
        high = high.asscalar()
    if shape == ():
        return _1_dim_array_to_flat_array(
            mx.nd.random.uniform(low, high, (1,), ctx=device, dtype=dtype)
        )
    return mx.nd.random.uniform(low, high, shape, ctx=device, dtype=dtype)


def random_normal(
    *,
    mean: Union[float, mx.nd.NDArray] = 0.0,
    std: Union[float, mx.nd.NDArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: mx.context.Context,
    dtype: type,
) -> mx.nd.NDArray:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    if isinstance(mean, mx.nd.NDArray):
        mean = mean.asscalar()
    if isinstance(std, mx.nd.NDArray):
        std = std.asscalar()
    if shape == ():
        return _1_dim_array_to_flat_array(
            mx.nd.random.normal(mean, std, (1,), ctx=device, dtype=dtype)
        )
    return mx.nd.random.normal(mean, std, shape, ctx=device, dtype=dtype)


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[mx.nd.NDArray] = None,
    replace: bool = True,
    device: mx.context.Context,
) -> mx.nd.NDArray:
    if not replace:
        raise Exception("MXNet does not support multinomial without replacement")
    if probs is None:
        probs = (
            mx.nd.ones(
                (
                    batch_size,
                    population_size,
                ),
                ctx=device,
            )
            / population_size
        )
    probs = probs / mx.nd.sum(probs, -1, True)
    return mx.nd.sample_multinomial(probs, (num_samples,))


def randint(
    low: Union[float, mx.nd.NDArray],
    high: Union[float, mx.nd.NDArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: mx.context.Context,
    dtype: Optional[Union[type, ivy.Dtype]] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape)
    if isinstance(low, mx.nd.NDArray):
        low = int(low.asscalar())
    if isinstance(high, mx.nd.NDArray):
        high = int(high.asscalar())
    if shape == ():
        return _1_dim_array_to_flat_array(
            mx.nd.random.randint(low, high, (1,), ctx=device, dtype=dtype)
        )
    return mx.nd.random.randint(low, high, shape, ctx=device, dtype=dtype)


def seed(*, seed_value: int = 0) -> None:
    mx.random.seed(seed_value)


def shuffle(x: mx.nd.NDArray, /) -> mx.nd.NDArray:
    return mx.nd.random.shuffle(x)
