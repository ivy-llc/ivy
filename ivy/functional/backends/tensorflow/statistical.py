# global
_round = round
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Tuple, Union, Optional


# Array API Standard #
# -------------------#

def min(x: Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> Tensor:
    return tf.math.reduce_min(x, axis = axis, keepdims = keepdims)


def sum(x: Tensor,
        axis: Optional[Union[int,Tuple[int]]] = None,
        dtype: Optional[tf.DType] = None,
        keepdims: bool = False) -> Tensor:

    if dtype == None:
        if x.dtype in [tf.int8, tf.int16, tf.int32]:
            dtype = tf.int32
        elif x.dtype in [tf.uint8, tf.uint16, tf.experimental.numpy.uint32]:
            dtype = tf.experimental.numpy.uint32
        elif x.dtype == tf.int64:
            dtype = tf.int64
        elif x.dtype == tf.uint64:
            dtype = tf.uint64

    return tf.experimental.numpy.sum(x, axis, dtype, keepdims)


def prod(x: Tensor,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[tf.DType] = None,
         keepdims: bool = False)\
        -> Tensor:
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


def max(x: Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> Tensor:
    return tf.math.reduce_max(x, axis = axis, keepdims = keepdims)

  
def var(x: Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> Tensor:
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    return tf.reduce_mean(tf.square(x - m), axis=axis, keepdims=keepdims)


def std(x: Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False)\
        -> Tensor:
    return tf.experimental.numpy.std(x, axis, keepdims)

    
# Extra #
# ------#

def einsum(equation, *operands):
    return tf.einsum(equation, *operands)
