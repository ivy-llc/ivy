# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple, List
import jaxlib.xla_extension
from jax.dlpack import from_dlpack as jax_from_dlpack

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_dev
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype


# Array API Standard #
# -------------------#


def ones(
    shape: Union[int, Tuple[int], List[int]],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return _to_dev(
        jnp.ones(shape, as_native_dtype(default_dtype(dtype))), device=device
    )


def zeros(
    shape: Union[int, Tuple[int], List[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
) -> JaxArray:
    return _to_dev(
        jnp.zeros(shape, dtype),
        device=device,
    )


def full_like(
    x: JaxArray,
    fill_value: Union[int, float],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_dev(
        jnp.full_like(
            x, fill_value, dtype=as_native_dtype(default_dtype(dtype, fill_value))
        ),
        device=device,
    )


def ones_like(
    x: JaxArray,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:

    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype
    return _to_dev(jnp.ones_like(x, dtype=dtype), device=device)


def zeros_like(
    x: JaxArray,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    if not dtype:
        dtype = x.dtype
    return _to_dev(jnp.zeros_like(x, dtype=dtype), device=device)


def tril(x: JaxArray, k: int = 0) -> JaxArray:
    return jnp.tril(x, k)


def triu(x: JaxArray, k: int = 0) -> JaxArray:
    return jnp.triu(x, k)


def empty(
    shape: Union[int, Tuple[int], List[int]],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return _to_dev(
        jnp.empty(shape, as_native_dtype(default_dtype(dtype))), device=device
    )


def empty_like(
    x: JaxArray,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:

    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_dev(jnp.empty_like(x, dtype=dtype), device=device)


def asarray(
    object_in,
    *,
    copy: Optional[bool] = None,
    dtype: jnp.dtype, 
    device: jaxlib.xla_extension.Device
):
    if copy is False:
        raise ValueError
    if isinstance(object_in, ivy.NativeArray) and dtype != "bool":
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        dtype = default_dtype(item=object_in, as_native=True)
        if copy is True:
            return _to_dev(jnp.array(object_in, dtype=dtype, copy=True), device=device)
        else:
            return _to_dev(jnp.asarray(object_in, dtype=dtype), device=device)
    else:
        dtype = default_dtype(dtype, object_in)

    if copy is True:
        return _to_dev(jnp.array(object_in, dtype=dtype, copy=True), device=device)
    else:
        return _to_dev(jnp.asarray(object_in, dtype=dtype), device=device)


def linspace(start, stop, num, axis=None, device=None, dtype=None, endpoint=True):
    if axis is None:
        axis = -1
    ans = jnp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    if dtype is None:
        ans = jnp.float32(ans)
    return _to_dev(ans, device=device)


def meshgrid(*arrays: JaxArray, indexing: str = "xy") -> List[JaxArray]:
    return jnp.meshgrid(*arrays, indexing=indexing)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    dtype = as_native_dtype(default_dtype(dtype))
    device = default_device(device)
    return _to_dev(jnp.eye(n_rows, n_cols, k, dtype), device=device)


# noinspection PyShadowingNames
def arange(
    start, stop=None, step=1, *, dtype: jnp.dtype, device: jaxlib.xla_extension.Device
):
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_dev(jnp.arange(start, stop, step=step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return _to_dev(
        jnp.full(shape, fill_value, as_native_dtype(default_dtype(dtype, fill_value))),
        device=device,
    )


def from_dlpack(x):
    return jax_from_dlpack(x)


# Extra #
# ------#

array = asarray


def logspace(start, stop, num, base=10.0, axis=None, device=None):
    if axis is None:
        axis = -1
    return _to_dev(jnp.logspace(start, stop, num, base=base, axis=axis), device=device)
