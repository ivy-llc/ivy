# global
from typing import Tuple
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def unique_inverse(x: JaxArray) \
        -> Tuple[JaxArray, JaxArray]:
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    return values, inverse_indices
