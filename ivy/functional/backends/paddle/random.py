"""Collection of Paddle random functions, wrapped to fit Ivy syntax and signature."""

# global
import paddle
from typing import Optional, Union, Sequence

# local
import ivy
from paddle.fluid.libpaddle import Place
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.backends.paddle.device import to_device
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.func_wrapper import with_unsupported_dtypes, with_unsupported_device_and_dtypes
from . import backend_version

# Extra #
# ------#


@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8",)},
    backend_version,
)
def random_uniform(
    *,
    low: Union[float, paddle.Tensor] = 0.0,
    high: Union[float, paddle.Tensor] = 1.0,
    shape: Optional[Union[paddle.Tensor, ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: Place,
    seed=None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    low = paddle.cast(low, "float32") if isinstance(low, paddle.Tensor) else low
    high = paddle.cast(high, "float32") if isinstance(high, paddle.Tensor) else high
    shape = _check_bounds_and_get_shape(low, high, shape)
    # Set range and seed
    rng = high - low
    if seed:
        _ = paddle.seed(seed)
    random_base = paddle.uniform(shape, min=0.0, max=1.0)
    with ivy.ArrayMode(False):
        return ivy.add(ivy.multiply(random_base, rng), low).cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def random_normal(
    *,
    mean: Union[float, paddle.Tensor] = 0.0,
    std: Union[float, paddle.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    if seed:
        paddle.seed(seed)
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        return paddle.normal(mean, std, shape).cast(dtype)
    if mean.dtype not in [paddle.float32, paddle.float64]:
        mean = mean.cast("float32")
    std = std.cast(mean.dtype)
    return paddle.normal(mean, std).cast(dtype)


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[paddle.Tensor] = None,
    replace: bool = True,
    device: Place,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8",)},
    backend_version,
)
def randint(
    low: Union[int, paddle.Tensor],
    high: Union[int, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Place,
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
    shape = _check_bounds_and_get_shape(low, high, shape)
    range = high - low
    if seed:
        _ = paddle.seed(seed)
    _retval = to_device(
        paddle.cast(
            paddle.uniform(shape or [1], min=0.0, max=1.0) * range + low, dtype
        ),
        device,
    )
    return _retval if shape else _retval.squeeze(axis=0)


def seed(*, seed_value: int = 0) -> None:
    _ = paddle.seed(seed_value)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def shuffle(
    x: paddle.Tensor,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if seed:
        _ = paddle.seed(seed)
    # Use Paddle's randperm function to generate shuffled indices
    indices = paddle.randperm(x.shape[0], dtype="int64")
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            shuffled_real = paddle.index_select(x.real(), indices)
            shuffled_imag = paddle.index_select(x.imag(), indices)
            return shuffled_real + 1j * shuffled_imag
        return paddle.index_select(x.cast("float32"), indices).cast(x.dtype)
    return paddle.index_select(x, indices)
