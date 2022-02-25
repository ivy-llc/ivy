# global
import jax.numpy as jnp


# local
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingBuiltins
def cross(x1: JaxArray, x2: JaxArray, /, *, axis: int = -1) -> JaxArray:
    return jnp.cross(a=x1, b=x2, axis=axis)
