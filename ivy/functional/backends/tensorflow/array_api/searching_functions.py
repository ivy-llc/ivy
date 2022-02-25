# global
import tensorflow as tf
from typing import Optional
from tensorflow.python.types.core import Tensor
from typing import Optional

def argmax(
    x: Tensor,
    axis: Optional[int] = None,
    out: Optional[int] = None
) -> Tensor:
    return tf.argmax(x, axis=axis, output_type=out)