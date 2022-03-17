# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def argsort(x: JaxArray,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True)\
            -> JaxArray:
    if descending:
        return jnp.asarray(jnp.argsort(-1 * jnp.searchsorted(jnp.unique(x), x), axis, kind='stable'))
    else:
        return jnp.asarray(jnp.argsort(x, axis, kind='stable'))
