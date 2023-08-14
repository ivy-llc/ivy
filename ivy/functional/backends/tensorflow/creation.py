# global
import numpy as np
from numbers import Number
from typing import Union, List, Optional, Sequence, Tuple

import tensorflow as tf

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_infer_dtype,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    asarray_inputs_to_native_shapes,
)
from . import backend_version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[tf.DType] = None,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    with tf.device(device):
        if dtype is None:
            if (
                isinstance(start, int)
                and isinstance(stop, int)
                and isinstance(step, int)
            ):
                return tf.cast(
                    tf.range(start, stop, delta=step, dtype=tf.int64), tf.int32
                )
            else:
                return tf.range(start, stop, delta=step)
        else:
            dtype = ivy.as_native_dtype(ivy.default_dtype(dtype=dtype))
            if dtype in [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.uint32, tf.uint64]:
                return tf.cast(tf.range(start, stop, delta=step, dtype=tf.int64), dtype)
            else:
                return tf.range(start, stop, delta=step, dtype=dtype)


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
@asarray_inputs_to_native_shapes
@asarray_infer_dtype
def asarray(
    obj: Union[
        tf.Tensor,
        tf.Variable,
        tf.TensorShape,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[tf.DType] = None,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        # convert the input to a tensor using the appropriate function
        try:
            ret = tf.convert_to_tensor(obj, dtype)
        except (TypeError, ValueError):
            ret = tf.cast(obj, dtype)
        return tf.identity(ret) if copy else ret


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.experimental.numpy.empty(shape, dtype)


def empty_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.experimental.numpy.empty_like(x, dtype=dtype)


@with_unsupported_dtypes({"2.13.0 and below": ("uint16",)}, backend_version)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        if n_cols is None:
            n_cols = n_rows
        if batch_shape is None:
            batch_shape = []
        i = tf.eye(n_rows, n_cols, dtype=dtype)
        reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
        tile_dims = list(batch_shape) + [1, 1]

        # k=index of the diagonal. A positive value refers to an upper diagonal,
        # a negative value to a lower diagonal, and 0 to the main diagonal.
        # Default: ``0``.
        # value of k ranges from -n_rows < k < n_cols

        # k=0 refers to the main diagonal
        if k == 0:
            return tf.eye(n_rows, n_cols, batch_shape=batch_shape, dtype=dtype)

        # when k is negative
        elif -n_rows < k < 0:
            mat = tf.concat(
                [tf.zeros([-k, n_cols], dtype=dtype), i[: n_rows + k]],
                0,
            )
            return tf.tile(tf.reshape(mat, reshape_dims), tile_dims)

        elif 0 < k < n_cols:
            mat = tf.concat(
                [
                    tf.zeros([n_rows, k], dtype=dtype),
                    i[:, : n_cols - k],
                ],
                1,
            )
            return tf.tile(tf.reshape(mat, reshape_dims), tile_dims)
        else:
            return tf.zeros(batch_shape + [n_rows, n_cols], dtype=dtype)


# noinspection PyShadowingNames
def from_dlpack(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(x, tf.Variable):
        x = x.read_value()
    dlcapsule = tf.experimental.dlpack.to_dlpack(x)
    return tf.experimental.dlpack.from_dlpack(dlcapsule)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, tf.DType]] = None,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    with tf.device(device):
        return tf.fill(
            shape,
            tf.constant(fill_value, dtype=dtype),
        )


def full_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    fill_value: Number,
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


def linspace(
    start: Union[tf.Tensor, tf.Variable, float],
    stop: Union[tf.Tensor, tf.Variable, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if axis is None:
        axis = -1
    with tf.device(device):
        start = tf.cast(tf.constant(start), dtype=dtype)
        stop = tf.cast(tf.constant(stop), dtype=dtype)
        if not endpoint:
            ans = tf.linspace(start, stop, num + 1, axis=axis)
            if axis < 0:
                axis += len(ans.shape)
            ans = tf.convert_to_tensor(
                ans.numpy()[_slice_at_axis(slice(None, -1), axis)]
            )
        else:
            ans = tf.linspace(start, stop, num, axis=axis)
        if dtype.is_integer and ans.dtype.is_floating:
            ans = tf.math.floor(ans)
        return tf.cast(ans, dtype)


@with_unsupported_dtypes({"2.13.0 and below": ("bool",)}, backend_version)
def meshgrid(
    *arrays: Union[tf.Tensor, tf.Variable],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    if not sparse:
        return tf.meshgrid(*arrays, indexing=indexing)

    sd = (1,) * len(arrays)
    res = [
        tf.reshape(tf.convert_to_tensor(a), (sd[:i] + (-1,) + sd[i + 1 :]))
        for i, a in enumerate(arrays)
    ]

    if indexing == "xy" and len(arrays) > 1:
        res[0] = tf.reshape(res[0], (1, -1) + sd[2:])
        res[1] = tf.reshape(res[1], (-1, 1) + sd[2:])

    return res


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.ones(shape, dtype)


def ones_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.ones_like(x, dtype=dtype)


@with_unsupported_dtypes({"2.13.0 and below": ("bool",)}, backend_version)
def tril(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # TODO: A way around tf.experimental.numpy.tril as it doesn't support bool
    #  and neither rank 1 tensors while np.tril does support both. Needs superset.
    return tf.experimental.numpy.tril(x, k)


@with_unsupported_dtypes({"2.13.0 and below": ("bool",)}, backend_version)
def triu(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.zeros(shape, dtype)


def zeros_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.zeros_like(x, dtype=dtype)


# Extra #
# ------#


array = asarray


def copy_array(
    x: Union[tf.Tensor, tf.Variable],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if to_ivy_array:
        return ivy.to_ivy(tf.identity(x))
    return tf.identity(x)


def one_hot(
    indices: Union[tf.Tensor, tf.Variable],
    depth: int,
    /,
    *,
    on_value: Number = 1,
    off_value: Number = 0,
    axis: Optional[int] = -1,
    dtype: Optional[tf.DType] = None,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    device = ivy.default_device(device)

    if device is not None:
        indices = tf.cast(indices, tf.int64)
        with tf.device(ivy.as_native_dev(device)):
            return tf.one_hot(
                indices,
                depth,
                on_value=on_value,
                off_value=off_value,
                axis=axis,
                dtype=dtype,
            )

    return tf.one_hot(
        indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype
    )


@with_unsupported_dtypes({"2.13.0 and below": ("uint32", "uint64")}, backend_version)
def frombuffer(
    buffer: bytes,
    dtype: Optional[tf.DType] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(buffer, bytearray):
        buffer = bytes(buffer)
    ret = tf.io.decode_raw(buffer, dtype)
    dtype = tf.dtypes.as_dtype(dtype)
    if offset > 0:
        offset = int(offset / dtype.size)
    if count > -1:
        ret = ret[offset : offset + count]
    else:
        ret = ret[offset:]

    return ret


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: str,
) -> Tuple[Union[tf.Tensor, tf.Variable]]:
    n_cols = n_rows if n_cols is None else n_cols

    if n_rows < 0 or n_cols < 0:
        n_rows, n_cols = 0, 0

    ret = [[], []]

    for i in range(0, min(n_rows, n_cols - k), 1):
        for j in range(max(0, k + i), n_cols, 1):
            ret[0].append(i)
            ret[1].append(j)

    if device is not None:
        with tf.device(ivy.as_native_dev(device)):
            return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))

    return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))
