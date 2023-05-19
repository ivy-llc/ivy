# global
from numbers import Number
import numpy as np
from typing import Union, Optional, List, Sequence

import jax.dlpack
import jax.numpy as jnp
import jaxlib.xla_extension

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)


# Array API Standard #
# ------------------ #


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
        ivy.utils.assertions._check_jax_x64_flag(dtype.name)
    res = _to_device(jnp.arange(start, stop, step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        JaxArray,
        bool,
        int,
        float,
        tuple,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(obj, ivy.NativeArray) and not dtype:
        if copy is True:
            dtype = obj.dtype
            ivy.utils.assertions._check_jax_x64_flag(dtype)
            return _to_device(jnp.array(obj, dtype=dtype, copy=True), device=device)
        else:
            return _to_device(obj, device=device)
    elif isinstance(obj, (list, tuple, dict)) and len(obj) != 0 and dtype is None:
        dtype = ivy.default_dtype(item=obj, as_native=True)
        ivy.utils.assertions._check_jax_x64_flag(dtype)
        if copy is True:
            return _to_device(jnp.array(obj, dtype=dtype, copy=True), device=device)
        else:
            return _to_device(jnp.asarray(obj, dtype=dtype), device=device)
    elif isinstance(obj, np.ndarray):
        dtype = ivy.as_native_dtype(ivy.as_ivy_dtype(obj.dtype.name))
        ivy.utils.assertions._check_jax_x64_flag(dtype)
    else:
        dtype = ivy.default_dtype(dtype=dtype, item=obj)
        ivy.utils.assertions._check_jax_x64_flag(dtype)

    if copy is True:
        return _to_device(jnp.array(obj, dtype=dtype, copy=True), device=device)
    else:
        return _to_device(jnp.asarray(obj, dtype=dtype), device=device)


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
    k: int = 0,
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
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(
        jnp.full(shape, fill_value, dtype),
        device=device,
    )


def full_like(
    x: JaxArray,
    /,
    fill_value: Number,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
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
        raise ivy.utils.exceptions.IvyException(
            f"Number of samples, {num}, must be non-negative."
        )

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


def meshgrid(
    *arrays: JaxArray,
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[JaxArray] = None,
) -> List[JaxArray]:
    return jnp.meshgrid(*arrays, sparse=sparse, indexing=indexing)


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


def copy_array(
    x: JaxArray, *, to_ivy_array: bool = True, out: Optional[JaxArray] = None
) -> JaxArray:
    x = (
        jax.core.ShapedArray(x.shape, x.dtype)
        if isinstance(x, jax.core.ShapedArray)
        else jnp.array(x)
    )
    if to_ivy_array:
        return ivy.to_ivy(x)
    return x


def one_hot(
    indices: JaxArray,
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    on_none = on_value is None
    off_none = off_value is None

    if dtype is None:
        if on_none and off_none:
            dtype = jnp.float32
        else:
            if not on_none:
                dtype = jnp.array(on_value).dtype
            elif not off_none:
                dtype = jnp.array(off_value).dtype

    res = jnp.eye(depth, dtype=dtype)[jnp.array(indices, dtype="int64").reshape(-1)]
    res = res.reshape(list(indices.shape) + [depth])

    if not on_none and not off_none:
        res = jnp.where(res == 1, on_value, off_value)

    if axis is not None:
        res = jnp.moveaxis(res, -1, axis)

    return _to_device(res, device)


def frombuffer(
    buffer: bytes,
    dtype: Optional[jnp.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> JaxArray:
    return jnp.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
