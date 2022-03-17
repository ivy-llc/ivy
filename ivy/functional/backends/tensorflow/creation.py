# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple, Optional
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy.functional.backends.tensorflow import Dtype
from ivy import dev_from_str, default_device, dtype_from_str, default_dtype


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


asarray = array
