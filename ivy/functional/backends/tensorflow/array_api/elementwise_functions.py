# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

def bitwise_and(x1: Tensor, x2: Tensor) -> Tensor:
    return tf.bitwise.bitwise_and(x1, x2)
