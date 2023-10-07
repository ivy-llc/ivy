# global
from typing import Optional
from typing import Tuple

import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray

# local


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> Tuple[JaxArray]:
    return jnp.unravel_index(indices.astype(jnp.int32), shape)
