# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing  import Union, Optional, Tuple, Literal

# local
inf = float("inf")

def vector_norm(x: Tensor, 
                p:Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int, ...]]] = None, 
                keepdims: bool = False)\
                 -> Tensor:

    tn_normalized_vector_ = None

    if p == -float('inf'):
        tn_normalized_vector_ = tf.reduce_min(tf.abs(x), axis, keepdims)
    elif p == -1:
        tn_normalized_vector_ = tf.reduce_sum(tf.abs(x)**p, axis, keepdims)**(1./p)

    elif p == 0:
        tn_normalized_vector_ = tf.reduce_sum(tf.cast(x != 0, 'float32'), axis, keepdims).numpy()

    else:
        tn_normalized_vector_ = tf.linalg.norm(x,p,axis,keepdims)

    if tn_normalized_vector_.shape  == tuple():
        return  tf.expand_dims(tn_normalized_vector_, 0)
    return tn_normalized_vector_