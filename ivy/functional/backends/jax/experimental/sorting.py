# global
import jax.numpy as jnp
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version


# invert_permutation
def invert_permutation(
    x: Union[JaxArray, list, tuple],
    /,
) -> JaxArray:
    x = jnp.array(x) if not ivy.is_array(x) else x
    sorted_indices = jnp.argsort(x)
    inverse = jnp.zeros_like(sorted_indices)
    inverse = inverse.at[sorted_indices].set(jnp.arange(len(x)))
    inverse_permutation = jnp.argsort(inverse)
    return inverse_permutation


# lexsort
@with_unsupported_dtypes({"0.4.24 and below": ("bfloat16",)}, backend_version)
def lexsort(
    keys: JaxArray,
    /,
    *,
    axis: int = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.asarray(jnp.lexsort(keys, axis=axis))
