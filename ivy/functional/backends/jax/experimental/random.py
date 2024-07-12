# global
from typing import Optional, Union, Sequence
import jax.numpy as jnp
import jax
import jaxlib.xla_extension

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.random import RNG, _setRNG, _getRNG  # noqa
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version

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
    device: Optional[jaxlib.xla_extension.Device] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(a, b, shape).shape
    RNG_, rng_input = jax.random.split(_getRNG())
    _setRNG(RNG_)
    if seed is not None:
        jax.random.PRNGKey(seed)
    return jax.random.beta(rng_input, a, b, shape, dtype)


@with_unsupported_dtypes({"0.4.24 and below": ("bfloat16",)}, backend_version)
def gamma(
    alpha: Union[float, JaxArray],
    beta: Union[float, JaxArray],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[jaxlib.xla_extension.Device] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    RNG_, rng_input = jax.random.split(_getRNG())
    _setRNG(RNG_)
    if seed is not None:
        jax.random.PRNGKey(seed)
    return jax.random.gamma(rng_input, alpha, shape, dtype) / beta


def poisson(
    lam: Union[float, JaxArray],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[jaxlib.xla_extension.Device] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    lam = jnp.array(lam)
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    if shape is not None:
        shape = jnp.array(shape)
        list_shape = shape.tolist()
        _check_shapes_broadcastable(lam.shape, list_shape)
    else:
        list_shape = None
    if jnp.any(lam < 0):
        pos_lam = jnp.where(lam < 0, 0, lam)
        ret = jax.random.poisson(rng_input, pos_lam, shape=list_shape).astype(dtype)
        ret = jnp.where(lam < 0, fill_value, ret)
    else:
        ret = jax.random.poisson(rng_input, lam, shape=list_shape).astype(dtype)
    return ret


def bernoulli(
    probs: Union[float, JaxArray],
    *,
    logits: Optional[Union[float, JaxArray]] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: Optional[jaxlib.xla_extension.Device] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = dtype if dtype is not None else probs.dtype
    if seed:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    if logits is not None:
        probs = jax.nn.softmax(logits, axis=-1)
    if hasattr(probs, "shape") and not _check_shapes_broadcastable(shape, probs.shape):
        shape = probs.shape
    return jax.random.bernoulli(rng_input, probs, shape=shape).astype(dtype)
