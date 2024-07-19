import tensorflow
import tensorflow as tf

from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_max


@tensorflow_handle_array_like_without_promotion
def tensorflow_softmax(
    x: tf.Tensor, /, *, axis: Optional[int] = None, out: Optional[tf.Tensor] = None
):
    if axis is None:
        axis = -1
    dtype = x.dtype
    if "complex" in str(dtype):
        amax = tensorflow_max(x, axis=axis, keepdims=True)
        normalized = tensorflow.exp(tensorflow.subtract(x, amax))
        return tensorflow.divide(
            normalized, tensorflow.reduce_sum(normalized, axis=axis, keepdims=True)
        )
    return tensorflow.nn.softmax(x, axis)
