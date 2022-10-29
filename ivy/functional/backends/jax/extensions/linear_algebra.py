from typing import Optional
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray


def diagflat(
    x: JaxArray,
    /,
    *,
    k: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.diagflat(x, k=k)
