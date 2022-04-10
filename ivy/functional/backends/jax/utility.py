# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingBuiltins
def all(x: JaxArray,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False,
        out: Optional[JaxArray]=None)\
        -> JaxArray:
    return jnp.all(x, axis, keepdims=keepdims, out=out)


# noinspection PyShadowingBuiltins
def any(x: JaxArray,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False)\
        -> JaxArray:
    return jnp.any(x, axis, keepdims=keepdims)
