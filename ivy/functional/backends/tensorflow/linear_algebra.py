# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
from ivy import inf
import ivy
from collections import namedtuple


inv = tf.linalg.inv
pinv = tf.linalg.pinv


def matrix_transpose(x: Tensor)\
        -> Tensor:
    return tf.experimental.numpy.swapaxes(x, -1, -2)


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


def qr(x: tf.Tensor,
       mode: str = 'reduced') -> namedtuple('qr', ['Q', 'R']):
    res = namedtuple('qr', ['Q', 'R'])
    if mode == 'reduced':
        q, r = tf.linalg.qr(x, full_matrices=False)
        return res(q, r)
    elif mode == 'complete':
        q, r = tf.linalg.qr(x, full_matrices=True)
        return res(q, r)
    else:
        raise Exception("Only 'reduced' and 'complete' qr modes are allowed for the tensorflow backend.")


def matmul(x1: tf.Tensor,
           x2: tf.Tensor) -> tf.Tensor:
    dtype_from = tf.experimental.numpy.promote_types(x1.dtype.as_numpy_dtype, x2.dtype.as_numpy_dtype)
    dtype_from = tf.as_dtype(dtype_from)
    if dtype_from.is_unsigned or dtype_from==tf.int8 or dtype_from==tf.int16:
        x1 = tf.cast(x1, tf.int64)
        x2 = tf.cast(x2, tf.int64)
    if x1.dtype != x2.dtype:
        x1 = tf.cast(x1, dtype_from)
        x2 = tf.cast(x2, dtype_from)

    if (x1.shape == () or x2.shape == ()
            or (len(x1.shape) == len(x2.shape) == 1 and x1.shape != x2.shape)
            or (len(x1.shape) == len(x2.shape) == 1 and x1.shape != x2.shape)
            or (len(x1.shape) == 1 and len(x2.shape) >= 2 and x1.shape[0] != x2.shape[-2])
            or (len(x2.shape) == 1 and len(x1.shape) >= 2 and x2.shape[0] != x1.shape[-1])
            or (len(x1.shape) >= 2 and len(x2.shape) >= 2 and x1.shape[-1] != x2.shape[-2])):
        raise Exception('Error,shapes not compatible')

    if len(x1.shape) == len(x2.shape) == 1:
        if x1.shape == 0:
            ret = tf.constant(0)
        else:

            ret = tf.math.multiply(x1, x2)[0]
        ret = tf.cast(ret, dtype=dtype_from)
        return ret

    x1_padded = False
    x1_padded_2 = False
    x2_padded = False

    if len(x1.shape) == 1:
        if len(x2.shape) == 2:
            x1_padded_2 = True
        elif len(x2.shape) > 2:
            x1_padded = True
        x1 = tf.expand_dims(x1, axis=0)

    elif len(x2.shape) == 1 and len(x1.shape) >= 2:
        x2 = tf.expand_dims(x2, axis=1)
        x2_padded = True

    ret = tf.matmul(x1, x2)

    ret = tf.cast(ret, dtype=dtype_from)
    if x1_padded_2:
        return ret[0]
    elif x1_padded:
        return tf.squeeze(ret, axis=-2)
    elif x2_padded:
        return tf.squeeze(ret, axis=-1)

    return ret


def svdvals(x: tf.Tensor) -> tf.Tensor:
    return tf.linalg.svd(x, compute_uv=False)


def slogdet(x:Union[ivy.Array,ivy.NativeArray],full_matrices: bool = True) -> Union[ivy.Array, Tuple[ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = tf.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res


def trace(x: tf.Tensor,
          offset: int = 0)\
              -> tf.Tensor:
    return tf.trace(x, offset)
