from typing import Optional
import tensorflow as tf


def difference(
    x1: tf.Tensor,
    x2: tf.Tensor = None,
    /,
    *,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    return tf.sets.difference(x1, x2)
