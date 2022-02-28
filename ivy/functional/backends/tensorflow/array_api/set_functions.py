# global
import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple
from collections import namedtuple
from tensorflow.python.ops.numpy_ops import np_config

def unique_counts(x: Tensor) \
                -> Tuple[Tensor, Tensor]:
    np_config.enable_numpy_behavior()
    uc = namedtuple('uc', ['values', 'counts'])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)