import tensorflow as tf
from typing import Optional
from tensorflow.python.types.core import Tensor

def argmin(
    x: Tensor,
    axis: Tensor = None,
    output_type: Optional[int] = tf.dtypes.int64,
) -> Tensor:

    ret = tf.math.argmin(x, axis=axis, output_type=output_type)
    return ret
