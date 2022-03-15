# global
from typing import Union, Tuple
import tensorflow as tf


def unique_inverse(x: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    values, inverse_indices = tf.unique(x)
    return values, inverse_indices
