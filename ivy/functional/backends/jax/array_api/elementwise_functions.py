# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray

def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)


def equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return x1 == x2


def less_equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return x1 <= x2
