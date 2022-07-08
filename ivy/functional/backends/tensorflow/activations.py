"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local


def linear(x: Tensor) -> Tensor:
    return tf.keras.activations.linear(x)


def hard_sigmoid(x: Tensor) -> Tensor:
    return tf.keras.activations.hard_sigmoid(x)


def relu(x: Tensor) -> Tensor:
    return tf.nn.relu(x)


def leaky_relu(x: Tensor, alpha: Optional[float] = 0.2) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


def gelu(x: Tensor, approximate: Optional[bool] = True) -> Tensor:
    return tf.nn.gelu(x, approximate)


def sigmoid(x: Tensor) -> Tensor:
    return tf.nn.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    return tf.nn.tanh(x)


def softmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis, keepdims=True)


def softplus(x: Tensor) -> Tensor:
    return tf.nn.softplus(x)
