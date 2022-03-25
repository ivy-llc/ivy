# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Tuple
from collections import namedtuple


def unique_inverse(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    x_1D = tf.reshape(x, [-1])
    values, inverse_indices = tf.unique(x_1D)
    if x.shape == tf.TensorShape([]):
        inverse_indices = tf.reshape(inverse_indices, [])
    return out(values, inverse_indices)


def unique_values(x: Tensor) \
        -> Tensor:
    return tf.unique(tf.reshape(x, [-1]))[0]
