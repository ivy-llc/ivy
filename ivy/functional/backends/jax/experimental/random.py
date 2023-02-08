# global
from typing import Optional, Union, Sequence
import jax.numpy as jnp
import jax
import jaxlib.xla_extension

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.random import RNG, _setRNG, _getRNG  # noqa
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)
from ivy.functional.backends.jax.device import to_device

# Extra #
# ----- #


# dirichlet
def dirichlet(
    alpha: Union[JaxArray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if seed is not None:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return jax.random.dirichlet(rng_input, alpha, shape=size, dtype=dtype)


def beta(
    a: Union[float, JaxArray],
    b: Union[float, JaxArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device = None,
    dtype: jnp.dtype = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(a, b, shape)
    RNG_, rng_input = jax.random.split(_getRNG())
    _setRNG(RNG_)
    if seed is not None:
        jax.random.PRNGKey(seed)
    return to_device(jax.random.beta(rng_input, a, b, shape, dtype), device)


def gamma(
    alpha: Union[float, JaxArray],
    beta: Union[float, JaxArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device = None,
    dtype: jnp.dtype = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape)
    RNG_, rng_input = jax.random.split(_getRNG())
    _setRNG(RNG_)
    if seed is not None:
        jax.random.PRNGKey(seed)
    return to_device(jax.random.gamma(rng_input, alpha, beta, shape, dtype), device)


@with_unsupported_dtypes({"0.3.14 and below": ("bfloat16",)}, backend_version)
def poisson(
    lam: Union[float, JaxArray],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    lam = jnp.array(lam)
    _check_shapes_broadcastable(shape, lam.shape)
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return to_device(
        jax.random.poisson(rng_input, lam, shape=shape),
        device,
    )


def bernoulli(
    probs: Union[float, JaxArray],
    *,
    logits: Union[float, JaxArray] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: jaxlib.xla_extension.Device,
    dtype: jnp.dtype = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    if logits is not None:
        probs = jax.nn.softmax(logits, axis=-1)
    if not _check_shapes_broadcastable(shape, probs.shape):
        shape = probs.shape
    return to_device(jax.random.bernoulli(rng_input, probs, shape=shape), device)
