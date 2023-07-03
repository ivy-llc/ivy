import jax.numpy as jnp
from typing import Tuple

# local
from ivy.functional.backends.jax import JaxArray


def intersection(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[JaxArray, JaxArray, JaxArray]:
    return jnp.intersect1d(
        x1, x2, assume_unique=assume_unique, return_indices=return_indices
    )
