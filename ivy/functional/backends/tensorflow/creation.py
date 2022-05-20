# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple, List, Optional
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy import (
    dev_from_str,
    default_device,
    dtype_from_str,
    default_dtype,
    dtype_to_str,
)


# Array API Standard #
# -------------------#


def asarray(object_in, dtype=None, device=None, copy=None):
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        if copy:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return tf.identity(object_in)
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    return tf.identity(tf.convert_to_tensor(object_in))
                except (TypeError, ValueError):
                    dtype = dtype_to_str(default_dtype(dtype, object_in))
                    return tf.identity(
                        tf.convert_to_tensor(
                            ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                            dtype=dtype,
                        )
                    )
            else:
                dtype = dtype_to_str(default_dtype(dtype, object_in))
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
                    dtype = dtype_to_str(default_dtype(dtype, object_in))
                    return tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
            else:
                dtype = dtype_to_str(default_dtype(dtype, object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.cast(tensor, dtype)


def zeros(
    shape: Union[int, Tuple[int]],
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[str] = None,
) -> Tensor:
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        return tf.zeros(shape, dtype_from_str(default_dtype(dtype)))


def ones(
    shape: Union[int, Tuple[int]],
    dtype: Optional[DType] = None,
    device: Optional[str] = None,
) -> tf.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    device = dev_from_str(default_device(device))
    with tf.device(device):
        return tf.ones(shape, dtype)


def full_like(
    x: Tensor,
    fill_value: Union[int, float],
    dtype: Optional[Union[DType, str, None]] = None,
    device: Optional[str] = None,
) -> Tensor:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = dev_from_str(default_device(device))
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def ones_like(
    x: Tensor,
    dtype: Optional[Union[DType, str, None]] = None,
    device: Optional[str] = None,
) -> Tensor:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        return tf.ones_like(x, dtype=dtype)


def zeros_like(
    x: Tensor,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[str] = None,
) -> Tensor:
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        return tf.zeros_like(x, dtype=dtype)


def tril(x: tf.Tensor, k: int = 0) -> tf.Tensor:
    return tf.experimental.numpy.tril(x, k)


def triu(x: tf.Tensor, k: int = 0) -> tf.Tensor:
    return tf.experimental.numpy.triu(x, k)


def empty(
    shape: Union[int, Tuple[int]],
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[str] = None,
) -> Tensor:
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        return tf.experimental.numpy.empty(shape, dtype_from_str(default_dtype(dtype)))


def empty_like(
    x: Tensor,
    dtype: Optional[Union[DType, str, None]] = None,
    device: Optional[str] = None,
) -> Tensor:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = default_device(device)
    with tf.device(dev_from_str(device)):
        return tf.experimental.numpy.empty_like(x, dtype=dtype)


def linspace(start, stop, num, axis=None, device=None, dtype=None, endpoint=True):
    if axis is None:
        axis = -1
    device = default_device(device)
    with tf.device(ivy.dev_from_str(device)):
        start = tf.constant(start, dtype=dtype)
        stop = tf.constant(stop, dtype=dtype)
        if not endpoint:
            ans = tf.linspace(start, stop, num + 1, axis=axis)[:-1]
        else:
            ans = tf.linspace(start, stop, num, axis=axis)
        if dtype is None:
            dtype = tf.float32
        ans = tf.cast(ans, dtype)
        return ans


def meshgrid(*arrays: tf.Tensor, indexing: str = "xy") -> List[tf.Tensor]:
    return tf.meshgrid(*arrays, indexing=indexing)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[str] = None,
) -> tf.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    device = dev_from_str(default_device(device))
    with tf.device(device):
        if n_cols is None:
            n_cols = n_rows
        i = tf.eye(n_rows, n_cols, dtype=dtype)
        if k == 0:
            return i
        elif -n_rows < k < 0:
            return tf.concat([tf.zeros([-k, n_cols], dtype=dtype), i[: n_rows + k]], 0)
        elif 0 < k < n_cols:
            return tf.concat(
                [tf.zeros([n_rows, k], dtype=dtype), i[:, : n_cols - k]], 1
            )
        else:
            return tf.zeros([n_rows, n_cols], dtype=dtype)


# noinspection PyShadowingNames
def arange(start, stop=None, step=1, dtype=None, device=None):

    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start

    device = dev_from_str(default_device(device))
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
            dtype = dtype_from_str(default_dtype(dtype))
            if dtype in [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.uint32, tf.uint64]:
                return tf.cast(tf.range(start, stop, delta=step, dtype=tf.int64), dtype)
            else:
                return tf.range(start, stop, delta=step, dtype=dtype)


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[str] = None,
) -> Tensor:
    with tf.device(dev_from_str(default_device(device))):
        return tf.fill(
            shape,
            tf.constant(
                fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value))
            ),
        )


def from_dlpack(x):
    return tf.experimental.dlpack.from_dlpack(x)


# Extra #
# ------#

array = asarray


def logspace(start, stop, num, base=10.0, axis=None, device=None):
    power_seq = linspace(start, stop, num, axis, default_device(device))
    return base**power_seq
