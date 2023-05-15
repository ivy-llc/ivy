from typing import Optional
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray

def union(
    x1: JaxArray,
    x2: JaxArray = None,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.setunion1d(
        x1,
        x2,
    )