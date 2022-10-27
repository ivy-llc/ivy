from typing import Optional, Union, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import jax
import ivy
from ivy.func_wrapper import with_supported_dtypes
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


# dirichlet
@with_supported_dtypes({"0.3.14 and below": ("float32", "float64")}, backend_version)
def dirichlet(
    alpha: Union[JaxArray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[jnp.dtype] = None,
    seed: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    RNG, rng_input = jax.random.split(_getRNG())
    if seed is not None:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return jax.random.dirichlet(rng_input, alpha, shape=size, dtype=dtype)
