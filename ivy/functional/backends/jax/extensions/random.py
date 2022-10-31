from typing import Optional, Union, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import jax
import ivy
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version

from ivy.functional.backends.jax.random import RNG, _setRNG, _getRNG  # noqa

# Extra #
# ----- #


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
    if seed is not None:
        rng_input = jax.random.PRNGKey(seed)
    else:
        RNG_, rng_input = jax.random.split(_getRNG())
        _setRNG(RNG_)
    return jax.random.dirichlet(rng_input, alpha, shape=size, dtype=dtype)
