import tensorflow as tf
from typing import Union


def l2_normalize(
    x: Union[tf.Tensor, tf.Variable], /, *, axis: int = None, out=None
) -> tf.Tensor:

    denorm = tf.norm(x, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)
