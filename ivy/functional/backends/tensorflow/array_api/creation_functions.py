# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple, Optional
from tensorflow.python.framework.dtypes import DType

# local
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


# noinspection SpellCheckingInspection
def full_like(x: Tensor, /,
              fill_value: Union[int, float], *,
              dtype: Optional[Union[DType, str, None]] = None,
              device: Optional[str] = None) \
        -> Tensor:
    device = dev_from_str(default_device(device))
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def tril(x: tf.Tensor,
         k: int = 0) \
        -> tf.Tensor:
    return tf.experimental.numpy.tril(x, k)
