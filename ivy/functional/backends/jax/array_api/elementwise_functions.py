# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)


def cos(x: JaxArray)\
        -> JaxArray:
    return jnp.cos(x)


def logical_not(x: JaxArray) -> JaxArray:
    return jnp.logical_not(x)
