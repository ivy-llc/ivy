# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple, List
import jaxlib.xla_extension
from jax.dlpack import from_dlpack as jax_from_dlpack

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import to_dev
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype


# Array API Standard #
# -------------------#


def ones(
    shape: Union[int, Tuple[int], List[int]],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return to_dev(
        jnp.ones(shape, as_native_dtype(default_dtype(dtype))), default_device(device)
    )


def zeros(
    shape: Union[int, Tuple[int], List[int]],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return to_dev(
        jnp.zeros(shape, default_dtype(dtype, as_str=False)),
        default_device(device, as_str=False),
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

    return to_dev(
        jnp.full_like(
            x, fill_value, dtype=as_native_dtype(default_dtype(dtype, fill_value))
        ),
        default_device(device),
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
    return to_dev(jnp.ones_like(x, dtype=dtype), default_device(device))


def zeros_like(
    x: JaxArray,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    if not dtype:
        dtype = x.dtype
    return to_dev(jnp.zeros_like(x, dtype=dtype), default_device(device))


def tril(x: JaxArray, k: int = 0) -> JaxArray:
    return jnp.tril(x, k)


def triu(x: JaxArray, k: int = 0) -> JaxArray:
    return jnp.triu(x, k)


def empty(
    shape: Union[int, Tuple[int], List[int]],
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
) -> JaxArray:
    return to_dev(
        jnp.empty(shape, as_native_dtype(default_dtype(dtype))), default_device(device)
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

    return to_dev(jnp.empty_like(x, dtype=dtype), default_device(device))


def asarray(
    object_in,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    copy: Optional[bool] = None,
):
    if isinstance(object_in, ivy.NativeArray) and dtype != "bool":
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        # Temporary fix on type
        # Because default_type() didn't return correct type for normal python array
        if copy is True:
            return to_dev(jnp.array(object_in, copy=True), device)
        else:
            return to_dev(jnp.asarray(object_in), device)
    else:
        dtype = default_dtype(dtype, object_in)
    if copy is True:
        return to_dev(jnp.array(object_in, dtype=dtype, copy=True), device)
    else:
        return to_dev(jnp.asarray(object_in, dtype=dtype), device)


def linspace(start, stop, num, axis=None, device=None, dtype=None, endpoint=True):
    if axis is None:
        axis = -1
    ans = jnp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    if dtype is None:
        ans = jnp.float32(ans)
    return to_dev(ans, device)


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
    return to_dev(jnp.eye(n_rows, n_cols, k, dtype), device)


# noinspection PyShadowingNames
def arange(start, stop=None, step=1, dtype=None, device=None):
    if dtype:
        dtype = as_native_dtype(dtype)
    res = to_dev(jnp.arange(start, stop, step=step, dtype=dtype), device)
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
    return to_dev(
        jnp.full(shape, fill_value, as_native_dtype(default_dtype(dtype, fill_value))),
        default_device(device),
    )


def from_dlpack(x):
    return jax_from_dlpack(x)


# Extra #
# ------#

array = asarray


def logspace(start, stop, num, base=10.0, axis=None, device=None):
    if axis is None:
        axis = -1
    return to_dev(
        jnp.logspace(start, stop, num, base=base, axis=axis), default_device(device)
    )
