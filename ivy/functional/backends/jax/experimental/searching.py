# global
import jax.numpy as jnp
from typing import Optional, Tuple

# local
from ivy.functional.backends.jax import JaxArray


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> Tuple[JaxArray]:
    return jnp.unravel_index(jnp.astype(indices, jnp.int32), shape)
