# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor


# local
import ivy


def isfinite(x: Tensor)\
        -> Tensor:
    if ivy.is_int_dtype(x):
        return tf.ones_like(x, tf.bool)
    return tf.math.is_finite(x)


def logical_not(x: Tensor) -> Tensor:
    return tf.logical_not(tf.cast(x, tf.bool))

def negative(x: Tensor) -> Tensor:
    if x.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.cast(tf.negative(tf.cast(x, tf.float32)), x.dtype)
    return tf.negative(x)
