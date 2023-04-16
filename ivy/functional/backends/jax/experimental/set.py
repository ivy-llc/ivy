from typing import Optional
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray


def difference(
    x1: JaxArray,
    x2: JaxArray = None,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.setdiff1d(
        x1,
        x2,
    )
