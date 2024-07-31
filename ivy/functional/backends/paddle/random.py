"""Collection of Paddle random functions, wrapped to fit Ivy syntax and
signature."""

# global
import paddle
import ivy.functional.backends.paddle as paddle_backend
from typing import Optional, Union, Sequence

# local
import ivy
from paddle.device import core
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_supported_device_and_dtypes,
    with_unsupported_dtypes,
)
from . import backend_version

# Extra #
# ------#


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("int8",)}},
    backend_version,
)
def random_uniform(
    *,
    low: Union[float, paddle.Tensor] = 0.0,
    high: Union[float, paddle.Tensor, None] = 1.0,
    shape: Optional[Union[paddle.Tensor, ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: core.Place = None,
    seed=None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if high is None:
        # default to float32, as this is the tf standard
        high = (
            paddle.finfo(dtype).max
            if dtype is not None
            else paddle.finfo(paddle.float32).max
        )
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    low = paddle.cast(low, "float32") if isinstance(low, paddle.Tensor) else low
    high = paddle.cast(high, "float32") if isinstance(high, paddle.Tensor) else high
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    # Set range and seed
    rng = high - low
    if seed:
        _ = paddle.seed(seed)
    random_base = paddle.uniform(shape, min=0.0, max=1.0)

    return paddle_backend.add(paddle_backend.multiply(random_base, rng), low).cast(
        dtype
    )


@with_unsupported_dtypes(
    {"2.6.0 and below": ("float16", "int16", "int8")}, backend_version
)
def random_normal(
    *,
    mean: Union[float, paddle.Tensor] = 0.0,
    std: Union[float, paddle.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape
    if seed:
        paddle.seed(seed)
    return paddle.normal(mean, std, shape).cast(dtype)


@with_supported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": (
                "float32",
                "float64",
            )
        }
    },
    backend_version,
)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[paddle.Tensor] = None,
    replace: bool = True,
    device: core.Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if probs is None:
        probs = paddle.ones((batch_size, num_samples)) / population_size
        probs = paddle.cast(probs, paddle.float32)
    if seed:
        paddle.seed(seed)
    x = paddle.multinomial(probs, num_samples=num_samples, replacement=replace)
    return x


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("int8",)}},
    backend_version,
)
def randint(
    low: Union[int, paddle.Tensor],
    high: Union[int, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: core.Place = None,
    dtype: Optional[Union[paddle.dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    low = paddle.cast(low, "float32") if isinstance(low, paddle.Tensor) else low
    high = paddle.cast(high, "float32") if isinstance(high, paddle.Tensor) else high
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    range = high - low
    if seed:
        _ = paddle.seed(seed)

    _retval = paddle.cast(
        paddle.uniform(shape or [1], min=0.0, max=1.0) * range + low, dtype
    )
    return _retval if shape else _retval.squeeze(axis=0)


def seed(*, seed_value: int = 0):
    _ = paddle.seed(seed_value)
    return


def shuffle(
    x: paddle.Tensor,
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if seed:
        _ = paddle.seed(seed)
    # Use Paddle's randperm function to generate shuffled indices
    indices = paddle.randperm(x.ndim, dtype="int64")
    if paddle.is_complex(x):
        shuffled_real = paddle.index_select(x.real(), indices, axis=axis)
        shuffled_imag = paddle.index_select(x.imag(), indices, axis=axis)
        return paddle.complex(shuffled_real, shuffled_imag)
    return paddle.index_select(x, indices, axis=axis)
