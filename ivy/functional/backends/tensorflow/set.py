import tensorflow as tf
from tensorflow.python.types.core import Tensor


def unique_values(x: Tensor) \
        -> Tensor:
    return tf.unique(tf.reshape(x, [-1]))[0]
