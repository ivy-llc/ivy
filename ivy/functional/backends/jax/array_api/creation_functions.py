# global
import jax
import jaxlib
import jax.numpy as jnp
from jaxlib.xla_extension import Buffer
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.functional.backends.jax import dtype_from_str
from ivy.functional.backends.jax.core.device import to_dev
from ivy.functional.ivy.core import default_device, default_dtype

JaxArray = Union[jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[jnp.dtype] = 'float32',
         device: Optional[str] = None) \
        -> JaxArray:
    return to_dev(jnp.ones(shape, dtype_from_str(default_dtype(dtype))), default_device(device))