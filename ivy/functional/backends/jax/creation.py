# For Review
# global
from typing import Union, Optional, Tuple, List, Sequence

import jax.dlpack
import jax.numpy as jnp
import jaxlib.xla_extension

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device
from ivy.functional.ivy import default_dtype

# noinspection PyProtectedMember
from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible


# Array API Standard #
# -------------------#


def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(jnp.arange(start, stop, step=step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res


def asarray(
    object_in: Union[JaxArray, jnp.ndarray, List[float], Tuple[float]],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(object_in, ivy.NativeArray) and not dtype:
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        dtype = default_dtype(item=object_in, as_native=True)
        if copy is True:
            return _to_device(
                jnp.array(object_in, dtype=dtype, copy=True), device=device
            )
        else:
            return _to_device(jnp.asarray(object_in, dtype=dtype), device=device)
    else:
        dtype = default_dtype(dtype=dtype, item=object_in)

    if copy is True:
        return _to_device(jnp.array(object_in, dtype=dtype, copy=True), device=device)
    else:
        return _to_device(jnp.asarray(object_in, dtype=dtype), device=device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(jnp.empty(shape, dtype), device=device)


def empty_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(jnp.empty_like(x, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: Optional[int] = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if n_cols is None:
        n_cols = n_rows
    i = jnp.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return _to_device(i, device=device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    return_mat = jnp.tile(jnp.reshape(i, reshape_dims), tile_dims)
    return _to_device(return_mat, device=device)


# noinspection PyShadowingNames
def from_dlpack(x, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    capsule = jax.dlpack.to_dlpack(x)
    return jax.dlpack.from_dlpack(capsule)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    return _to_device(
        jnp.full(shape, fill_value, dtype),
        device=device,
    )


def full_like(
    x: JaxArray,
    /,
    fill_value: float,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    return _to_device(
        jnp.full_like(x, fill_value, dtype=dtype),
        device=device,
    )


# https://github.com/google/jax/blob/8b2e4f975c8c830502f5cc749b7253b02e78c9e8/jax/_src/numpy/lax_numpy.py#L2164
# with some modification
def linspace(
    start: Union[JaxArray, float],
    stop: float,
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        axis = -1

    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")

    if dtype is None:
        dtype = ivy.promote_types(start.dtype, stop.dtype)
    dtype = jnp.dtype(dtype)
    computation_dtype = dtype
    start = jnp.asarray(start, dtype=computation_dtype)
    stop = jnp.asarray(stop, dtype=computation_dtype)

    bounds_shape = list(jax.lax.broadcast_shapes(jnp.shape(start), jnp.shape(stop)))
    broadcast_start = jnp.broadcast_to(start, bounds_shape)
    broadcast_stop = jnp.broadcast_to(stop, bounds_shape)
    axis = len(bounds_shape) + axis + 1 if axis < 0 else axis
    bounds_shape.insert(axis, 1)
    div = (num - 1) if endpoint else num
    if num > 1:
        iota_shape = [
            1,
        ] * len(bounds_shape)
        iota_shape[axis] = div
        # This approach recovers the endpoints with float32 arithmetic,
        # but can lead to rounding errors for integer outputs.
        real_dtype = jnp.finfo(computation_dtype).dtype
        step = jnp.reshape(jax.lax.iota(real_dtype, div), iota_shape) / div
        step = step.astype(computation_dtype)
        start_reshaped = jnp.reshape(broadcast_start, bounds_shape)
        end_reshaped = jnp.reshape(broadcast_stop, bounds_shape)
        out = start_reshaped + step * (end_reshaped - start_reshaped)

        if endpoint:
            out = jax.lax.concatenate(
                [out, jax.lax.expand_dims(broadcast_stop, (axis,))],
                jax._src.util.canonicalize_axis(axis, out.ndim),
            )

    elif num == 1:
        out = jnp.reshape(broadcast_start, bounds_shape)
    else:  # num == 0 degenerate case, match numpy behavior
        empty_shape = list(jax.lax.broadcast_shapes(jnp.shape(start), jnp.shape(stop)))
        empty_shape.insert(axis, 0)
        out = jnp.reshape(jnp.array([], dtype=dtype), empty_shape)

    if jnp.issubdtype(dtype, jnp.integer) and not jnp.issubdtype(
        out.dtype, jnp.integer
    ):
        out = jax.lax.floor(out)

    ans = jax.lax.convert_element_type(out, dtype)

    return _to_device(ans, device=device)


def meshgrid(*arrays: JaxArray, indexing: str = "xy") -> List[JaxArray]:
    return jnp.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(jnp.ones(shape, dtype), device=device)


def ones_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(jnp.ones_like(x, dtype=dtype), device=device)


def tril(x: JaxArray, /, *, k: int = 0, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tril(x, k)


def triu(x: JaxArray, /, *, k: int = 0, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(
        jnp.zeros(shape, dtype),
        device=device,
    )


def zeros_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return _to_device(jnp.zeros_like(x, dtype=dtype), device=device)


# Extra #
# ------#


array = asarray


def logspace(
    start: Union[JaxArray, int],
    stop: Union[JaxArray, int],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: int = None,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        axis = -1
    return _to_device(
        jnp.logspace(start, stop, num, base=base, dtype=dtype, axis=axis), device=device
    )
