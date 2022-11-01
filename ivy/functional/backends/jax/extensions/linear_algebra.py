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


def kron(
    a: JaxArray,
    b: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.kron(a, b)
