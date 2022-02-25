# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


def roll(x: JaxArray, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None) -> JaxArray:
    return jnp.roll(x, shift, axis)


# noinspection PyShadowingBuiltins
def flip(x: JaxArray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> JaxArray:
    return jnp.flip(x, axis=axis)
