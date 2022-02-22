# global
import jax
import jaxlib
import jax.numpy as jnp
from typing import Union
from jaxlib.xla_extension import Buffer

JaxArray = Union[jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]

def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)
