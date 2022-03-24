# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor


def argsort(x: Tensor,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True) \
        -> Tensor:
    if tf.convert_to_tensor(x).dtype.is_bool:
        if descending:
            return tf.argsort(tf.cast(x, dtype=tf.int32), axis=axis, direction='DESCENDING', stable=stable)
        else:
            return tf.argsort(tf.cast(x, dtype=tf.int32), axis=axis, direction='ASCENDING', stable=stable)
    else:
        if descending:
            return tf.argsort(tf.convert_to_tensor(x), axis=axis, direction='DESCENDING', stable=stable)
        else:
            return tf.argsort(tf.convert_to_tensor(x), axis=axis, direction='ASCENDING', stable=stable)


def sort(x: tf.Tensor,
         axis: int = -1,
         descending: bool = False,
         stable: bool = True) \
        -> tf.Tensor:
    if tf.convert_to_tensor(x).dtype.is_bool:
        if descending:
            res = tf.sort(tf.cast(x, dtype=tf.int32), axis=axis, direction='DESCENDING')
            return tf.cast(res, tf.bool)
        else:
            res = tf.sort(tf.cast(x, dtype=tf.int32), axis=axis, direction='ASCENDING')
            return tf.cast(res, tf.bool)
    else:
        if descending:
            return tf.sort(tf.convert_to_tensor(x), axis=axis, direction='DESCENDING')
        else:
            return tf.sort(tf.convert_to_tensor(x), axis=axis, direction='ASCENDING')