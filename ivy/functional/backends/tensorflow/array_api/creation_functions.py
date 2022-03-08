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


def linspace(start: Union[int, float],
             stop: Union[int, float],
             num: int,
             dtype: Optional[Dtype] = None,
             device: Optional[str] = None,
             endpoint: bool = True) \
             -> tf.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    dev = dev_from_str(default_device(device))
    with tf.device(dev):
        if endpoint:
            return tf.cast(tf.linspace(start, stop, num), dtype)
        else:
            return tf.cast(tf.linspace(start, stop, num+1), dtype)[:-1]

          
def tril(x: tf.Tensor,
         k: int = 0) \
         -> tf.Tensor:
    return tf.experimental.numpy.tril(x, k)

