#global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
#Local
import ivy

def isnan(x: Tensor)-> []:
    if ivy.is_int_dtype(x):
        return tf.zeros_like(x, tf.bool)
    return tf.math.is_nan(x)