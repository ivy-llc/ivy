"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy


def relu(
    x: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.nn.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(
    x: Union[tf.Tensor, tf.Variable],
    alpha: Optional[float] = 0.2,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.leaky_relu(x, alpha)


gelu = lambda x, approximate=True: tf.nn.gelu(x, approximate)


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
