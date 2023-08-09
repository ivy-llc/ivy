"""Collection of Jax random functions, wrapped to fit Ivy syntax and signature."""

# global
import jax
import jax.numpy as jnp
import jaxlib.xla_extension
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.functional.backends.jax import JaxArray
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Extra #
# ------#


class RNGWrapper:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)


RNG = RNGWrapper()


def _setRNG(key):
    global RNG
    RNG.key = key


def _getRNG():
    global RNG
    return RNG.key


def random_uniform(
    *,
    low: Union[float, JaxArray] = 0.0,
    high: Union[float, JaxArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(low, high, shape).shape

    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return jax.random.uniform(
            rng_input, shape, minval=low, maxval=high, dtype=jnp.float32
        ).astype(dtype)


def random_normal(
    *,
    mean: Union[float, JaxArray] = 0.0,
    std: Union[float, JaxArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape

    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return jax.random.normal(rng_input, shape, dtype=dtype) * std + mean


@with_unsupported_dtypes({"0.4.14 and below": ("bfloat16",)}, backend_version)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[JaxArray] = None,
    replace: bool = True,
    device: jaxlib.xla_extension.Device,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    RNG_, rng_input = jax.random.split(_getRNG())
    _setRNG(RNG_)
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)

    if probs is None:
        probs = (
            jnp.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = jnp.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / jnp.sum(probs_flat, -1, keepdims=True, dtype="float64")
    probs_stack = jnp.split(probs_flat, probs_flat.shape[0])
    samples_stack = [
        jax.random.choice(rng_input, num_classes, (num_samples,), replace, p=prob[0])
        for prob in probs_stack
    ]
    samples_flat = jnp.stack(samples_stack)
    return jnp.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples])


def randint(
    low: Union[int, JaxArray],
    high: Union[int, JaxArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: Optional[Union[jnp.dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape).shape

    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)

    return jax.random.randint(rng_input, shape, low, high, dtype)


def seed(*, seed_value: int = 0) -> None:
    _setRNG(jax.random.PRNGKey(seed_value))
    return


def shuffle(
    x: JaxArray,
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if x.shape == ():
        return x
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)

    # jax.random.shuffle is deprecated; identical behaviour reproduced with
    # jax.random.permutation
    return jax.random.permutation(key=rng_input, x=x, axis=axis, independent=True)
