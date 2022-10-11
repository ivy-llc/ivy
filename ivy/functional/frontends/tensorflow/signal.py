import tensorflow as tf

# hann_window


def hann_window(length, dtype=tf.float32):
    return tf.signal.hann_window(length=length, dtype=dtype)
