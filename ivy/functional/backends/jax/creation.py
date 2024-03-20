# global
from numbers import Number
import numpy as np
from typing import Union, Optional, List, Sequence, Tuple

import jax.dlpack
import jax.numpy as jnp
import jax._src as _src
import jaxlib.xla_extension

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import dev
from ivy.functional.ivy.creation import (
    _asarray_to_native_arrays_and_back,
    _asarray_infer_device,
    _asarray_infer_dtype,
    _asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    _asarray_inputs_to_native_shapes,
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
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if dtype:
        dtype = as_native_dtype(dtype)
        ivy.utils.assertions._check_jax_x64_flag(dtype.name)
    res = jnp.arange(start, stop, step, dtype=dtype)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res


@_asarray_to_native_arrays_and_back
@_asarray_infer_device
@_asarray_handle_nestable
@_asarray_inputs_to_native_shapes
@_asarray_infer_dtype
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
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions._check_jax_x64_flag(dtype)
    ret = jnp.asarray(obj, dtype=dtype)
    # jnp.copy is used to ensure correct device placement
    # it's slower than jax.device_put before JIT, but it's necessary to use since
    # jax device objects aren't serializable and prevent saving transpiled graphs
    # this workaround only works because we are inside jax.default_device context
    # invoked in @handle_device decorator
    return jnp.copy(ret) if (dev(ret, as_native=True) != device or copy) else ret


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.empty(shape, dtype)


def empty_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.empty_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if n_cols is None:
        n_cols = n_rows
    i = jnp.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return i
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    return_mat = jnp.tile(jnp.reshape(i, reshape_dims), tile_dims)
    return return_mat


def to_dlpack(x, /, *, out: Optional[JaxArray] = None):
    return jax.dlpack.to_dlpack(x)


def from_dlpack(x, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.dlpack.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    return jnp.full(shape, fill_value, dtype)


def full_like(
    x: JaxArray,
    /,
    fill_value: Number,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.full_like(x, fill_value, dtype=dtype)


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
    device: jaxlib.xla_extension.Device = None,
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
                _src.util.canonicalize_axis(axis, out.ndim),
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

    return ans


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
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.ones(shape, dtype)


def ones_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.ones_like(x, dtype=dtype)


def tril(x: JaxArray, /, *, k: int = 0, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tril(x, k)


def triu(x: JaxArray, /, *, k: int = 0, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.zeros(shape, dtype)


def zeros_like(
    x: JaxArray,
    /,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.zeros_like(x, dtype=dtype)


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
    device: jaxlib.xla_extension.Device = None,
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

    return res


def frombuffer(
    buffer: bytes,
    dtype: Optional[jnp.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> JaxArray:
    return jnp.frombuffer(buffer, dtype=dtype, count=count, offset=offset)


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: jaxlib.xla_extension.Device = None,
) -> Tuple[JaxArray]:
    return jnp.triu_indices(n=n_rows, k=k, m=n_cols)
