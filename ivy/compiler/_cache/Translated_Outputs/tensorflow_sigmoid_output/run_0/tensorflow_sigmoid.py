import tensorflow
import tensorflow as tf

from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_sigmoid(
    x: tf.Tensor, /, *, complex_mode="jax", out: Optional[tf.Tensor] = None
):
    return 1 / (1 + tensorflow.exp(-x))
