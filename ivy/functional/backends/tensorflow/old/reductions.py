"""
Collection of TensorFlow reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf


def einsum(equation, *operands):
    return _tf.einsum(equation, *operands)
