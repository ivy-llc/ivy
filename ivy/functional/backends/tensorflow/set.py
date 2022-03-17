# global
from typing import Union, Tuple
import tensorflow as tf
from tensorflow.python.types.core import Tensor


def unique_inverse(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    values, inverse_indices = tf.unique(x)
    return values, inverse_indices
