"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local


def relu(x: Tensor) -> Tensor:
    return tf.nn.relu(x)


def leaky_relu(x: Tensor, alpha: Optional[float] = 0.2) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


def gelu(x: Tensor, approximate: bool =True)\
    -> Tensor:
    return tf.nn.gelu(x, approximate)


def sigmoid(x: Tensor) -> Tensor:
    return tf.nn.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    return tf.nn.tanh(x)


def softmax(x: Tensor, axis: Optional[int] = -1) -> Tensor:
    return tf.nn.softmax(x, axis)


def softplus(x: Tensor) -> Tensor:
    return tf.nn.softplus(x)
