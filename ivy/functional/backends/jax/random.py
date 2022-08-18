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
from ivy.functional.backends.jax.device import to_device

# Extra #
# ------#

RNG = jax.random.PRNGKey(0)


def random_uniform(
    *,
    low: Union[float, JaxArray] = 0.0,
    high: Union[float, JaxArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(low, high, shape)
    global RNG
    RNG, rng_input = jax.random.split(RNG)
    return to_device(
        jax.random.uniform(rng_input, shape, minval=low, maxval=high, dtype=dtype),
        device,
    )


def random_normal(
    *,
    mean: Union[float, JaxArray] = 0.0,
    std: Union[float, JaxArray] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    global RNG
    RNG, rng_input = jax.random.split(RNG)
    return (
        to_device(
            jax.random.normal(rng_input, shape, dtype=dtype),
            device,
        )
        * std
        + mean
    )


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[JaxArray] = None,
    replace: bool = True,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    global RNG
    RNG, rng_input = jax.random.split(RNG)
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
    return to_device(
        jnp.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]),
        device,
    )


def randint(
    low: Union[int, JaxArray],
    high: Union[int, JaxArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: Optional[Union[jnp.dtype, ivy.Dtype]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape)
    global RNG
    RNG, rng_input = jax.random.split(RNG)
    return to_device(jax.random.randint(rng_input, shape, low, high, dtype), device)


def seed(*, seed_value: int = 0) -> None:
    global RNG
    RNG = jax.random.PRNGKey(seed_value)
    return


def shuffle(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    global RNG
    RNG, rng_input = jax.random.split(RNG)
    return jax.random.shuffle(rng_input, x)
