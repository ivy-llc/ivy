import tensorflow as tf
import ivy
from typing import Optional, Union


def logit(x: Union[tf.Tensor, tf.Variable],
          /,
          *,
          eps: Optional[float] = None,
          out=None):
    x_dtype = x.dtype
    if eps is None:
        x = tf.where(tf.math.logical_or(x > 1, x < 0), ivy.nan, x)
    else:
        x = tf.clip_by_value(x, eps, 1 - eps)
    return tf.cast(tf.math.log(x / (1 - x)), x_dtype)
