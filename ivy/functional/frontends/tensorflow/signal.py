import ivy
import ivy.functional.frontends.tensorflow as ivy_tf

# hann_window

def hann_window(window_length, periodic=True, dtype=ivy.int32, name=None):
    return ivy_tf.tensorflow.signal.hann_window(window_length, periodic, dtype)

