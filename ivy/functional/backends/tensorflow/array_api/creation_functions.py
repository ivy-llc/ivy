# global
import tensorflow as tf
from typing import Tuple, Optional, Union
from tensorflow.python.framework.dtypes import DType

# local
import ivy


# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[DType] = None,
         device: Optional[str] = None) \
        -> tf.Tensor:
    dtype = ivy.dtype_from_str(ivy.default_dtype(dtype))
    dev = ivy.dev_from_str(ivy.default_device(device))
    with tf.device(dev):
        return tf.ones(shape, dtype)
