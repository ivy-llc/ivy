# global
import jax
import jaxlib
import jax.numpy as jnp
from jaxlib.xla_extension import Buffer
from typing import Union, Tuple, Optional, List

JaxArray = Union[jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]


# noinspection PyShadowingBuiltins
def all(x: JaxArray,
        axis: Optional[Union[int, Tuple[int, ...], List[int, ...]]] = None,
        keepdims: bool = False)\
        -> JaxArray:
    return jnp.all(x, axis, keepdims=keepdims)
