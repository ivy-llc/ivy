# global
from typing import Optional, Union, Sequence
import jax.numpy as jnp
import jax
import jaxlib.xla_extension

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.random import RNG, _setRNG, _getRNG  # noqa
from ivy.functional.ivy.random import _check_bounds_and_get_shape
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
