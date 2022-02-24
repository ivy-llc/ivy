import tensorflow as tf
from tensorflow.python.types.core import Tensor

def argmax(
    x: Tensor,
    axis: Optional[int] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    return tf.argmax(x,axis=axis,output_type=out)