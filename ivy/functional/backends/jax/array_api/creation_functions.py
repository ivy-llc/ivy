# global
import jaxlib
import jax.numpy as jnp

from typing import Union, Optional, Tuple, List

# local
from ivy import dtype_from_str
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.core.device import to_dev
from ivy.functional.ivy.core import default_device, default_dtype
from jaxlib.xla_extension import Device


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


# noinspection SpellCheckingInspection
def full_like(x: JaxArray,
              fill_value: Union[int, float],
              dtype: Optional[jnp.dtype] = None,
              device: Optional[jaxlib.xla_extension.Device] = None) \
        -> JaxArray:
    if not dtype:
        dtype = x.dtype
    return to_dev(jnp.full_like(x, fill_value, dype=dtype_from_str(default_dtype(dtype, fill_value))),
                  default_device(device))


def tril(x: JaxArray,
         k: int = 0) \
        -> JaxArray:
    return jnp.tril(x, k)
