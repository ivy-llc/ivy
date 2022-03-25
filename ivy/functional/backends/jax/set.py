import jax
import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray


def unique_values(x: JaxArray) \
        -> JaxArray:
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if (nan_count > 1):
        unique = jnp.append(jnp.unique(x.flatten()), jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    else:
        unique = jnp.unique(x.flatten()).astype(x.dtype)
    return unique
