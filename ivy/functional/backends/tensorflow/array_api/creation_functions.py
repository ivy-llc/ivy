# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple

# local
from ivy.functional.backends.tensorflow import Dtype
from ivy import dev_from_str, default_device, dtype_from_str, default_dtype


# noinspection PyShadowingNames
def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: Dtype = None,
          device: str = None) \
        -> Tensor:
    dev = default_device(device)
    with tf.device(dev_from_str(dev)):
        return tf.zeros(shape, dtype_from_str(default_dtype(dtype)))