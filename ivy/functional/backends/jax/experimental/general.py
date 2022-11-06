
from typing import Optional
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def isin(
    elements: JaxArray,
    test_elements: JaxArray,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> JaxArray:
    return jnp.isin(elements, test_elements,
                    assume_unique=assume_unique,
                    invert=invert)
