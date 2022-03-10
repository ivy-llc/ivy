# global
import tensorflow as tf
from typing import Optional, Union
from tensorflow.python.types.core import Tensor
from typing import Optional

def argmax(
    x: Tensor,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[int] = tf.dtypes.int64,
) -> Tensor:
    
    ret = tf.constant(x).numpy().argmax(axis=axis, keepdims=keepdims)
    ret_dtype = ret.dtype
    ret = tf.convert_to_tensor(ret,dtype=ret_dtype)

    return ret