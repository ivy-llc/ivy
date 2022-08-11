# For Review
# global
import tensorflow as tf
from typing import Union, Tuple, List, Optional, Sequence

# local
import ivy
from ivy import (
    as_native_dtype,
    default_dtype,
    as_ivy_dtype,
)

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
    dtype: Optional[tf.DType] = None,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
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
            dtype = as_native_dtype(default_dtype(dtype=dtype))
            if dtype in [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.uint32, tf.uint64]:
                return tf.cast(tf.range(start, stop, delta=step, dtype=tf.int64), dtype)
            else:
                return tf.range(start, stop, delta=step, dtype=dtype)


def asarray(
    object_in: Union[tf.Tensor, tf.Variable, List[float], Tuple[float]],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: tf.DType = None,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        if copy:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return tf.identity(object_in)
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    dtype = default_dtype(item=object_in, as_native=True)
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    dtype = default_dtype(dtype=dtype, item=object_in, as_native=True)
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.identity(tf.cast(tensor, dtype))
            else:
                dtype = as_ivy_dtype(default_dtype(dtype=dtype, item=object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.identity(tf.cast(tensor, dtype))
        else:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return object_in
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    return tf.convert_to_tensor(object_in)
                except (TypeError, ValueError):
                    dtype = as_ivy_dtype(default_dtype(dtype=dtype, item=object_in))
                    return tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
            else:
                dtype = as_ivy_dtype(default_dtype(dtype=dtype, item=object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.cast(tensor, dtype)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.experimental.numpy.empty(shape, dtype)


def empty_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.experimental.numpy.empty_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: Optional[int] = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
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
        # Default: 0.
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


eye.unsupported_dtypes = ("uint16",)


# noinspection PyShadowingNames
def from_dlpack(
    x: Union[tf.Tensor, tf.Variable], /, *, out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    dlcapsule = tf.experimental.dlpack.to_dlpack(x)
    return tf.experimental.dlpack.from_dlpack(dlcapsule)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, tf.DType]] = None,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    with tf.device(device):
        return tf.fill(
            shape,
            tf.constant(fill_value, dtype=dtype),
        )


def full_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    fill_value: Union[int, float],
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


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
    out: Union[tf.Tensor, tf.Variable] = None
):
    if axis is None:
        axis = -1
    with tf.device(device):
        start = tf.constant(start, dtype=dtype)
        stop = tf.constant(stop, dtype=dtype)
        if not endpoint:
            ans = tf.linspace(start, stop, num + 1, axis=axis)[:-1]
        else:
            ans = tf.linspace(start, stop, num, axis=axis)
        ans = tf.cast(ans, dtype)
        return ans


def meshgrid(
    *arrays: Union[tf.Tensor, tf.Variable], indexing: str = "xy"
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.ones(shape, dtype)


def ones_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.ones_like(x, dtype=dtype)


def tril(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.tril(x, k)


def triu(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.zeros(shape, dtype)


def zeros_like(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.zeros_like(x, dtype=dtype)


# Extra #
# ------#


array = asarray


def logspace(
    start: Union[tf.Tensor, tf.Variable, int],
    stop: Union[tf.Tensor, tf.Variable, int],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: Optional[int] = None,
    dtype: tf.DType,
    device: str,
    out: Union[tf.Tensor, tf.Variable] = None
) -> Union[tf.Tensor, tf.Variable]:
    power_seq = ivy.linspace(start, stop, num, axis, dtype=dtype, device=device)
    return base**power_seq
