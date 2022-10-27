# global
from typing import Optional, Union

import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


# msort
def msort(
    a: Union[JaxArray, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.msort(a)
