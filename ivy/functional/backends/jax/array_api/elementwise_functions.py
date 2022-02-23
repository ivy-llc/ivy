# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray

def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)

def less(x1: JaxArray,x2:JaxArray)\
        -> JaxArray:
    return jnp.less(x1,x2)
