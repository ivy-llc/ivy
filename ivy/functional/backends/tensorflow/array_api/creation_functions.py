import tensorflow as tf


def zeros(shape, dtype=None) -> tf.Tensor:
    return tf.experimental.numpy.zeros(shape, dtype=dtype)

