# global
import tensorflow as tf
from typing import Tuple, Optional, Union

# local
from ivy.functional.backends.tensorflow import dtype_from_str, dev_from_str
from ivy.functional.ivy.core import default_device, default_dtype

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[tf.dtype] = 'float32',
         device: Optional[str] = None) \
        -> tf.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    dev = dev_from_str(default_device(device))
    with tf.device(dev):
        return tf.ones(shape, dtype)