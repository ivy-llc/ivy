# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple, Optional
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy.functional.backends.tensorflow import Dtype
from ivy import dev_from_str, default_device, dtype_from_str, default_dtype, dtype_to_str


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[Dtype] = None,
          device: Optional[str] = None) \
        -> Tensor:
    dev = default_device(device)
    with tf.device(dev_from_str(dev)):
        return tf.zeros(shape, dtype_from_str(default_dtype(dtype)))


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[DType] = None,
         device: Optional[str] = None) \
        -> tf.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    dev = dev_from_str(default_device(device))
    with tf.device(dev):
        return tf.ones(shape, dtype)


def full_like(x: Tensor,
              fill_value: Union[int, float],
              dtype: Optional[Union[DType, str, None]] = None,
              device: Optional[str] = None) \
        -> Tensor:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = dev_from_str(default_device(device))
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def ones_like(x : Tensor,
              dtype: Optional[Union[DType, str, None]] = None,
              dev: Optional[str] = None) \
        -> Tensor:
    dtype = tf.DType(dtype) if dtype is str else dtype
    dev = default_device(dev)
    with tf.device(dev_from_str(dev)):
        return tf.ones_like(x, dtype=dtype)


def tril(x: tf.Tensor,
         k: int = 0) \
         -> tf.Tensor:
    return tf.experimental.numpy.tril(x, k)


def triu(x: tf.Tensor,
         k: int = 0) \
         -> tf.Tensor:
    return tf.experimental.numpy.triu(x, k)
    
    
def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[Dtype] = None,
          device: Optional[str] = None) \
        -> Tensor:
    dev = default_device(device)
    with tf.device(dev_from_str(dev)):
        return tf.experimental.numpy.empty(shape, dtype_from_str(default_dtype(dtype)))


# Extra #
# ------#

# noinspection PyShadowingNames
def array(object_in, dtype=None, dev=None):
    dtype = dtype_from_str(default_dtype(dtype, object_in))
    dev = default_device(dev)
    with tf.device(dev_from_str(dev)):
        try:
            tensor = tf.convert_to_tensor(object_in, dtype=dtype)
        except (TypeError, ValueError):
            tensor = tf.convert_to_tensor(ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)), dtype=dtype)
        if dtype is None:
            return tensor
        return tf.cast(tensor, dtype)


def asarray(object_in, dtype=None, dev=None, copy=None):
    dev = default_device(dev)
    with tf.device(dev_from_str(dev)):
        if copy:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return tf.identity(object_in)
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    return tf.identity(tf.convert_to_tensor(object_in))
                except (TypeError, ValueError):
                    dtype = dtype_to_str(default_dtype(dtype, object_in))
                    return tf.identity(tf.convert_to_tensor(ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)), dtype=dtype))
            else:
                dtype = dtype_to_str(default_dtype(dtype, object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)), dtype=dtype)
                return tf.identity(tf.cast(tensor, dtype))
        else:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return object_in
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    return tf.convert_to_tensor(object_in)
                except (TypeError, ValueError):
                    dtype = dtype_to_str(default_dtype(dtype, object_in))
                    return tf.convert_to_tensor(ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)), dtype=dtype)
            else:
                dtype = dtype_to_str(default_dtype(dtype, object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)), dtype=dtype)
                return tf.cast(tensor, dtype)