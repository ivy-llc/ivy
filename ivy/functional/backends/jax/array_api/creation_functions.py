# global
import jax
import jaxlib
import jax.numpy as jnp
from jaxlib.xla_extension import Buffer
from typing import Union, Tuple

# local
from ivy import dtype_from_str
from ivy.functional.backends.jax.core.device import to_dev
from ivy.functional.ivy.core import default_device, default_dtype

JaxArray = Union[jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]


# noinspection PyShadowingNames
def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: jnp.dtype = None,
          device: jaxlib.xla_extension.Device = None) \
        -> JaxArray:
    return to_dev(jnp.zeros(shape, dtype_from_str(default_dtype(dtype))), default_device(device))
