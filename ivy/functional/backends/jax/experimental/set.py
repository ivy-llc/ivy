import jax.numpy as jnp
from typing import Tuple

# local
from ivy.functional.backends.jax import JaxArray


def intersection(
        ar1: JaxArray,
        ar2: JaxArray,
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
) -> Tuple[JaxArray, JaxArray]:
    return jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
