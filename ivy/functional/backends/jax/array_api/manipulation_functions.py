# gloabl
from typing import Union, Tuple
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def roll(x: JaxArray, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None) -> JaxArray:
    return jnp.roll(x, shift, axis)