# global
import jaxlib
import jax.numpy as jnp
from typing import Union, Optional, Tuple, List

# local
from ivy import dtype_from_str
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import to_dev
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype
from jaxlib.xla_extension import Buffer, Device, DeviceArray
from jax.interpreters.xla import _DeviceArray

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


def full_like(x: JaxArray,
              fill_value: Union[int, float],
              dtype: Optional[jnp.dtype] = None,
              device: Optional[jaxlib.xla_extension.Device] = None) \
        -> DeviceArray:
    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return to_dev(jnp.full_like(x, fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value))),
              default_device(device))


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


def empty_like(x: JaxArray,
              dtype: Optional[Union[jnp.dtype, str]]=None,
              dev: Optional[Union[Device, str]] = None)\
        -> DeviceArray:

    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return to_dev(jnp.empty_like(x, dtype=dtype), default_device(dev))


def asarray(object_in, dtype: Optional[str] = None, dev: Optional[str] = None, copy: Optional[bool] = None):
    if isinstance(object_in, (_DeviceArray, DeviceArray, Buffer)):
        dtype = object_in.dtype
    elif isinstance(object_in, (list, tuple, dict)) and len(object_in) != 0 and dtype is None:
        # Temporary fix on type
        # Because default_type() didn't return correct type for normal python array
        if copy is True:
            return to_dev(jnp.array(object_in,copy=True), dev)
        else:
            return to_dev(jnp.asarray(object_in), dev)
    else:
        dtype = default_dtype(dtype, object_in)
    if copy is True:
        return to_dev(jnp.array(object_in, dtype=dtype,copy=True), dev)
    else:
        return to_dev(jnp.asarray(object_in, dtype=dtype), dev)


def linspace(start, stop, num, axis=None, dev=None):
    if axis is None:
        axis = -1
    return to_dev(jnp.linspace(start, stop, num, axis=axis), default_device(dev))

def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[jnp.dtype] = None,
        device: Optional[jaxlib.xla_extension.Device] = None) \
        -> JaxArray:
    dtype = dtype_from_str(default_dtype(dtype))
    device = default_device(device)
    return to_dev(jnp.eye(n_rows, n_cols, k, dtype), device)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype=None, dev=None):
    dtype = dtype_from_str(dtype)
    return to_dev(jnp.arange(start, stop, step=step, dtype=dtype), default_device(dev))


# Extra #
# ------#

# noinspection PyShadowingNames
def array(object_in, dtype=None, dev=None):
    return to_dev(jnp.array(object_in, dtype=dtype_from_str(default_dtype(dtype, object_in))), default_device(dev))


def logspace(start, stop, num, base=10., axis=None, dev=None):
    if axis is None:
        axis = -1
    return to_dev(jnp.logspace(start, stop, num, base=base, axis=axis), default_device(dev))
