# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy


def bitwise_and(x1: Tensor,
                x2: Tensor)\
        -> Tensor:
    return tf.bitwise.bitwise_and(x1, x2)


def ceil(x: Tensor)\
        -> Tensor:
    if 'int' in str(x.dtype):
        return x
    return tf.math.ceil(x)


def isfinite(x: Tensor) \
        -> Tensor:
    if ivy.is_int_dtype(x):
        return tf.ones_like(x, tf.bool)
    return tf.math.is_finite(x)


def asinh(x: Tensor) \
        -> Tensor:
    return tf.asinh(x)


def sqrt(x: Tensor)\
        -> Tensor:
    return tf.math.sqrt(x)


def cosh(x: Tensor) \
        -> Tensor:
    return tf.cosh(x)


def log2(x: Tensor) \
        -> Tensor:
    return tf.experimental.numpy.log2(x)


def isnan(x: Tensor)\
        -> Tensor:
    if ivy.is_int_dtype(x):
        return tf.zeros_like(x, tf.bool)
    return tf.math.is_nan(x)


def less(x1: Tensor, x2: Tensor)\
        -> Tensor:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)
        x1 = tf.cast(x1, promoted_type)
        x2 = tf.cast(x2, promoted_type)
    return tf.math.less(x1, x2)


def cos(x: Tensor)\
        -> Tensor:
    return tf.cos(x)


def logical_not(x: Tensor)\
        -> Tensor:
    return tf.logical_not(tf.cast(x, tf.bool))
