# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List


# local
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingBuiltins
def cross(x1: jax.Array, x2: jax.Array, /, *, axis: int = -1) -> jax.Array:
    return jnp.cross(a, b, axis=axis)
