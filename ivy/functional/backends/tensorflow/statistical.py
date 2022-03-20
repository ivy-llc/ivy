# global
_round = round
import tensorflow as tf
from typing import Tuple, Union, Optional


# Array API Standard #
# -------------------#

def min(x: tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> tf.Tensor:
    return tf.math.reduce_min(x, axis = axis, keepdims = keepdims)


def sum(x: tf.Tensor,
               axis: Optional[Union[int,Tuple[int]]] = None,
               keepdims: bool = False)\
                   -> tf.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def prod(x: tf.Tensor,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[tf.DType] = None,
         keepdims: bool = False)\
        -> tf.Tensor:
    if dtype == None:
        if x.dtype in [ tf.int8 , tf.int16,tf.int32]:
            dtype = tf.int32
        elif x.dtype in [ tf.uint8,tf.uint16,tf.experimental.numpy.uint32]:
            dtype = tf.experimental.numpy.uint32
        elif x.dtype == tf.int64: 
            dtype = tf.int64
        elif x.dtype == tf.uint64 :
            dtype = tf.uint64
        
    return tf.experimental.numpy.prod(x,axis,dtype,keepdims)


def mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)


def max(x: tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> tf.Tensor:
    return tf.math.reduce_max(x, axis = axis, keepdims = keepdims)

    
# Extra #
# ------#
