# global
from typing import Tuple
import jax.numpy as jnp


def unique_inverse(x: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    return values, inverse_indices
