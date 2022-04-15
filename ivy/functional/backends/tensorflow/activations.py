"""
Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor


def relu(x: Tensor)\
        -> Tensor:
    return tf.nn.relu(x)


def leaky_relu(x: Tensor, alpha: Optional[float] = 0.2)\
        -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


gelu = lambda x, approximate=True: tf.nn.gelu(x, approximate)

def tanh(x: Tensor)\
        -> Tensor:
    return tf.nn.tanh(x)

sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
softplus = tf.nn.softplus
