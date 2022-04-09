# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Tuple
from collections import namedtuple


def unique_all(x: Tensor) \
    -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
    UniqueAllResult = namedtuple(typename='unique_all', field_names=['values', 'indices', 'inverse_indices', 'counts'])
    
    values, indices, counts = tf.unique_with_counts(x.flatten())
    _, inverse_indices = tf.unique(tf.reshape(x, -1))
    
    return UniqueAllResult(values.reshape(x.shape), indices, tf.reshape(inverse_indices, tf.shape(x)), counts)


def unique_inverse(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = tf.unique(tf.reshape(x, -1))
    inverse_indices = tf.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: Tensor) \
        -> Tensor:
    return tf.unique(tf.reshape(x, [-1]))[0]


def unique_counts(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)
