# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Optional, Tuple, Literal

# local
from ivy import inf


# noinspection PyUnusedLocal,PyShadowingBuiltins
def vector_norm(x: Tensor,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, - inf]] = 2)\
                 -> Tensor:

    if ord == -float('inf'):
        tn_normalized_vector = tf.reduce_min(tf.abs(x), axis, keepdims)
    elif ord == -1:
        tn_normalized_vector = tf.reduce_sum(tf.abs(x)**ord, axis, keepdims)**(1./ord)

    elif ord == 0:
        tn_normalized_vector = tf.reduce_sum(tf.cast(x != 0, 'float32'), axis, keepdims).numpy()

    else:
        tn_normalized_vector = tf.linalg.norm(x, ord, axis, keepdims)

    if tn_normalized_vector.shape == tuple():
        return tf.expand_dims(tn_normalized_vector, 0)
    return tn_normalized_vector


"""
def outer(x1: tf.Tensor,
          x2: tf.Tensor)\
        -> tf.Tensor:
    #M = tf.Tensor(x1, x2, axes=0)
    #Not sure if the code in the # will work
    return tf.experimental.numpy.outer(x1, x2)
"""
#This is my code that was requested to be fixed


def outer(x1: Tensor,
          x2: Tensor) \
        -> Tensor:
    return tf.experimental.numpy.outer(x1, x2)
#This is my fixed code but I am not sure if it works properly as i am not sure why the above code does not work

def diagonal(x: tf.Tensor,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> tf.Tensor:
    return tf.experimental.numpy.diagonal(x, offset, axis1=axis1, axis2=axis2)
