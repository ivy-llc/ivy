"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union

# global
import tensorflow as tf

# local


def relu(
    x: Union[tf.Tensor, tf.Variable],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.relu(x)


def leaky_relu(
    x: Union[tf.Tensor, tf.Variable],
    alpha: Optional[float] = 0.2,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.leaky_relu(x, alpha)


def gelu(x, approximate=True):
    return tf.nn.gelu(x, approximate)


def sigmoid(
    x: Union[tf.Tensor, tf.Variable],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.sigmoid(x)


def tanh(
    x: Union[tf.Tensor, tf.Variable],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.tanh(x)


def softmax(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = -1,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.softmax(x, axis)


def softplus(
    x: Union[tf.Tensor, tf.Variable],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.softplus(x)
