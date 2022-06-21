"""Collection of MXNet random functions, wrapped to fit Ivy syntax and signature."""

# global
import mxnet as mx
from typing import Optional, Union, Tuple, Sequence

# local
import ivy
from ivy.functional.ivy.device import default_device

# noinspection PyProtectedMember
from ivy.functional.backends.mxnet import _mxnet_init_context

# noinspection PyProtectedMember
from ivy.functional.backends.mxnet import _1_dim_array_to_flat_array


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
    dtype=None,
) -> mx.nd.NDArray:
    if isinstance(low, mx.nd.NDArray):
        low = low.asscalar()
    if isinstance(high, mx.nd.NDArray):
        high = high.asscalar()
    ctx = _mxnet_init_context(default_device(device))
    if shape is None or len(shape) == 0:
        return _1_dim_array_to_flat_array(
            mx.nd.random.uniform(low, high, (1,), ctx=ctx, dtype=dtype)
        )
    return mx.nd.random.uniform(low, high, shape, ctx=ctx, dtype=dtype)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    if isinstance(mean, mx.nd.NDArray):
        mean = mean.asscalar()
    if isinstance(std, mx.nd.NDArray):
        std = std.asscalar()
    ctx = _mxnet_init_context(default_device(device))
    if shape is None or len(shape) == 0:
        return _1_dim_array_to_flat_array(mx.nd.random.normal(mean, std, (1,), ctx=ctx))
    return mx.nd.random.uniform(mean, std, shape, ctx=ctx)


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[mx.nd.NDArray] = None,
    replace: bool = True,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    if not replace:
        raise Exception("MXNet does not support multinomial without replacement")
    ctx = _mxnet_init_context(default_device(device))
    if probs is None:
        probs = (
            mx.nd.ones(
                (
                    batch_size,
                    population_size,
                ),
                ctx=ctx,
            )
            / population_size
        )
    probs = probs / mx.nd.sum(probs, -1, True)
    return mx.nd.sample_multinomial(probs, (num_samples,))


def randint(
    low: int,
    high: int,
    shape: Union[int, Sequence[int]],
    *,
    device: mx.context.Context,
    out: Optional[mx.nd.NDArray],
) -> mx.nd.NDArray:
    if isinstance(low, mx.nd.NDArray):
        low = int(low.asscalar())
    if isinstance(high, mx.nd.NDArray):
        high = int(high.asscalar())
    ctx = _mxnet_init_context(default_device(device))
    if len(shape) == 0:
        return _1_dim_array_to_flat_array(
            mx.nd.random.randint(low, high, (1,), ctx=ctx)
        )
    return mx.nd.random.randint(low, high, shape, ctx=ctx)


def seed(seed_value: int = 0) -> None:
    mx.random.seed(seed_value)


def shuffle(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.random.shuffle(x)
