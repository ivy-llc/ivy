"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy


def gelu(
    x: Tensor, /, *, approximate: Optional[bool] = True, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.gelu(x, approximate)


def leaky_relu(
    x: Tensor, /, *, alpha: Optional[float] = 0.2, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


def relu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu(x)


def sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return tf.nn.sigmoid(x)


def softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.softmax(x, axis)


def softplus(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.softplus(x)
