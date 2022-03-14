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

def stack(x: Union[JaxArray, Tuple[JaxArray], List[JaxArray]],
          axis: Optional[int] = None)\
          -> JaxArray:
    if x is JaxArray:
        x = [x]
    if axis is None:
        axis = 0
    return jnp.stack(x, axis=axis)
