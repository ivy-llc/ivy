# global
import jax
import jaxlib
import jax.numpy as jnp
from typing import Union
from jaxlib.xla_extension import Buffer

# local
import ivy

JaxArray = Union[jax.interpreters.xla.DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]

def bitwise_and(x1: JaxArray, x2: JaxArray, /) -> JaxArray:
    return jnp.bitwise_and(x1, x2)