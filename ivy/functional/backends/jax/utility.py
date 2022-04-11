# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray
import ivy


# noinspection PyShadowingBuiltins
def all(x: JaxArray,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False,
        out: Optional[JaxArray]=None)\
        -> JaxArray:
    ret = jnp.all(x, axis, keepdims=keepdims)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# noinspection PyShadowingBuiltins
def any(x: JaxArray,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False,
        out: Optional[JaxArray]=None)\
        -> JaxArray:
    ret = jnp.any(x, axis, keepdims=keepdims)
    if ivy.exists(out):
        ivy.inplace_update(out,ret)
    return ret
