"""
Collection of TensorFlow logic functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

logical_and = lambda x1, x2: _tf.logical_and(_tf.cast(x1, _tf.bool), _tf.cast(x2, _tf.bool))
logical_or = lambda x1, x2: _tf.logical_or(_tf.cast(x1, _tf.bool), _tf.cast(x2, _tf.bool))
logical_not = lambda x: _tf.logical_not(_tf.cast(x, _tf.bool))
