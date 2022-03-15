# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingBuiltins
def flip(x: JaxArray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> JaxArray:
    return jnp.flip(x, axis=axis)

