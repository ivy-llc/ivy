# global
import jax
import jax.numpy as jnp
from typing import Tuple
from collections import namedtuple

# local
from ivy.functional.backends.jax import JaxArray


def unique_inverse(x: JaxArray) \
        -> Tuple[JaxArray, JaxArray]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    if x.shape == ():
        inverse_indices = inverse_indices.reshape(())
    return out(values, inverse_indices)


def unique_values(x: JaxArray) \
        -> JaxArray:
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if (nan_count > 1):
        unique = jnp.append(jnp.unique(x.flatten()), jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    else:
        unique = jnp.unique(x.flatten()).astype(x.dtype)
    return unique
