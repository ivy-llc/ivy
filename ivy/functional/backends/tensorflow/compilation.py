"""Collection of Tensorflow compilation functions."""

# global
import tensorflow as tf


def compile(
    fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    return tf.function(fn)
