# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray

def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)
