# global
import jax.numpy as jnp
from typing import Optional, Union

# local
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version


# msort
def msort(
    a: Union[JaxArray, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.msort(a)


# lexsort
@with_unsupported_dtypes({"0.3.14 and below": ("bfloat16",)}, backend_version)
def lexsort(
    keys: JaxArray,
    /,
    *,
    axis: int = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.asarray(jnp.lexsort(keys, axis=axis))
