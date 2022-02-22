from typing import Union

import jax.numpy as jnp
from jaxlib.xla_extension import Buffer
import jax
import jaxlib

JaxArray = Union[jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]


def zeros(shape, dtype=None) -> JaxArray:
    return jnp.zeros(shape, dtype=dtype)


#print(jnp.zeros([3, 4], jnp.int32))
