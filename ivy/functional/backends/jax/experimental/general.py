<<<<<<< HEAD
from typing import Optional
=======
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
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
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
) -> JaxArray:
    return jnp.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
