# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
from ivy import inf
import ivy as _ivy
from collections import namedtuple



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

# noinspection PyPep8Naming
def svd(x:Tensor,full_matrices: bool = True) -> Union[Tensor, Tuple[Tensor,...]]:
    results=namedtuple("svd", "U S Vh")

    batch_shape = tf.shape(x)[:-2]
    num_batch_dims = len(batch_shape)
    transpose_dims = list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims]
    D, U, V = tf.linalg.svd(x,full_matrices=full_matrices)
    VT = tf.transpose(V, transpose_dims)
    res=results(U, D, VT)
    return res

def diagonal(x: tf.Tensor,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> tf.Tensor:
    return tf.experimental.numpy.diagonal(x, offset, axis1=axis1, axis2=axis2)

def slogdet(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = tf.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res
