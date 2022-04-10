# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Tuple
from collections import namedtuple

import ivy


def unique_inverse(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = tf.unique(tf.reshape(x, -1))
    inverse_indices = tf.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: Tensor, out: Tensor = None) \
        -> Tensor:
    ret = tf.unique(tf.reshape(x, [-1]))[0]
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def unique_counts(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)
