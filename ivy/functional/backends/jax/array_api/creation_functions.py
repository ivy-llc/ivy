# global
import jaxlib
import jax.numpy as jnp
from typing import Union, Optional, Tuple, List

# local
from ivy import dtype_from_str
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.core.device import to_dev
from ivy.functional.ivy.core import default_device, default_dtype


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[jnp.dtype] = None,
         device: Optional[jaxlib.xla_extension.Device] = None) \
        -> JaxArray:
    return to_dev(jnp.ones(shape, dtype_from_str(default_dtype(dtype))), default_device(device))


def zeros(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[jnp.dtype] = None,
          device: Optional[jaxlib.xla_extension.Device] = None) \
        -> JaxArray:
    return to_dev(jnp.zeros(shape, dtype_from_str(default_dtype(dtype))), default_device(device))

def linspace(start: Union[int, float],
             stop: Union[int, float],
             num: int,
             dtype: Optional[jnp.dtype] = None,
             device: Optional[jaxlib.xla_extension.Device] = None,
             endpoint: bool = True) \
             -> JaxArray:
    if dtype is None:
        dtype = jnp.float32
    return to_dev(jnp.linspace(jnp.double(start), jnp.double(stop), num, dtype=dtype, endpoint=endpoint),
                  default_device(device))

