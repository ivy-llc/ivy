<<<<<<< HEAD
from typing import Optional
=======
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"0.3.14 and below": ("float16", "bfloat16")}, backend_version)
def isin(
    elements: JaxArray,
    test_elements: JaxArray,
    /,
    *,
<<<<<<< HEAD
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
=======
    assume_unique: bool = False,
    invert: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
) -> JaxArray:
    return jnp.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
