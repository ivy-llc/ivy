"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import pi
from tensorflow.python.types.core import Tensor

# local
import ivy


def relu(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: Tensor, alpha: Optional[float] = 0.2) -> Tensor:
    if tf.reduce_any(alpha>x):
        alpha_dtype = ivy.default_dtype(item=alpha)
        promoted_dtype = tf.experimental.numpy.promote_types(x.dtype,alpha_dtype)
        x = tf.cast(x,dtype=promoted_dtype)
    return tf.nn.leaky_relu(x, alpha)



def gelu (x: Tensor, approximate:Optional[bool]=True)->Tensor:
    return tf.nn.gelu(x,approximate)



def sigmoid(x: Tensor) -> Tensor:
    return tf.nn.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    return tf.nn.tanh(x)


def softmax(x: Tensor, axis: Optional[int] = -1) -> Tensor:
    return tf.nn.softmax(x, axis)


def softplus(x: Tensor) -> Tensor:
    return tf.nn.softplus(x)
