import tensorflow as tf


# hann_window 
def hann_window(window_length, dtype=None):
    return tf.signal.hann_window(window_length, dtype=dtype)
