# global
import tensorflow as tf
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor


def cross(x1: Tensor, x2: Tensor, /, *, axis: int = -1) -> Tensor:
    return tf.linalg.cross(a=x1, b=x2)
