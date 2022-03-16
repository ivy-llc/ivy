# global
import jaxlib
import jax.numpy as jnp
from typing import Union, Optional, Tuple, List

# local
from ivy import dtype_from_str
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import to_dev
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy.old import default_dtype
from jaxlib.xla_extension import Device, DeviceArray


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


def ones_like(x : JaxArray,
              dtype: Optional[Union[jnp.dtype, str]]=None,
              dev: Optional[Union[Device, str]] = None)\
        -> DeviceArray:

    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype
    return to_dev(jnp.ones_like(x, dtype=dtype), default_device(dev))


def tril(x: JaxArray,
         k: int = 0) \
         -> JaxArray:
    return jnp.tril(x, k)

  
def triu(x: JaxArray,
         k: int = 0) \
         -> JaxArray:
    return jnp.triu(x, k)
    

def empty(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[jnp.dtype] = None,
          device: Optional[jaxlib.xla_extension.Device] = None) \
        -> JaxArray:
    return to_dev(jnp.empty(shape, dtype_from_str(default_dtype(dtype))), default_device(device))


# Extra #
# ------#
# noinspection PyShadowingNames
def array(object_in, dtype=None, dev=None):
    return to_dev(jnp.array(object_in, dtype=dtype_from_str(default_dtype(dtype, object_in))), default_device(dev))


asarray = array