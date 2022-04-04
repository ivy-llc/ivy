# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Optional, Tuple, Literal, List
from collections import namedtuple

# local
from ivy import inf
import ivy



# Array API Standard #
# -------------------#

def eigh(x: Tensor)\
 -> Tensor:
        return tf.linalg.eigh(x) 


def inv(x: Tensor) -> Tensor:
    if tf.math.reduce_any(tf.linalg.det(x) == 0 ):
        return x
    return tf.linalg.inv(x)


def tensordot(x1: Tensor, x2: Tensor,
              axes: Union[int, Tuple[List[int], List[int]]] = 2) \
        -> Tensor:

    # find type to promote to
    dtype = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)

    # type casting to float32 which is acceptable for tf.tensordot
    x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)

    return tf.cast(tf.tensordot(x1, x2, axes), dtype)


def pinv(x: Tensor,
         rtol: Optional[Union[float, Tuple[float]]] = None) \
        -> Tensor:
    if rtol is None:
        return tf.linalg.pinv(x)
    return tf.linalg.pinv(tf.cast(x != 0, 'float32'), tf.cast(rtol != 0, 'float32'))


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


def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception('if specified, axes must be a length-2 sequence of ints,'
                        'but found {} of type {}'.format(axes, type(axes)))
    if p == -float('inf'):
        ret = tf.reduce_min(tf.reduce_sum(tf.abs(x), axis=axes[1], keepdims=True), axis=axes)
    elif p == -1:
        ret = tf.reduce_min(tf.reduce_sum(tf.abs(x), axis=axes[0], keepdims=True), axis=axes)
    else:
        ret = tf.linalg.norm(x, p, axes, keepdims)
    if ret.shape == ():
        return tf.expand_dims(ret, 0)
    return ret


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


def outer(x1:tf.Tensor,
          x2: tf.Tensor) \
        -> tf.Tensor:
    return tf.experimental.numpy.outer(x1, x2)


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


def det(x:tf.Tensor,name:Optional[str]=None) \
    -> tf.Tensor:
    return tf.linalg.det(x,name)

def cholesky(x: tf.Tensor,
            upper: bool = False) -> tf.Tensor:
    if not upper:
        return tf.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        return tf.transpose(tf.linalg.cholesky(tf.transpose(x, perm=axes)),
                            perm=axes)


def eigvalsh(x: Tensor) -> Tensor:
    return tf.linalg.eigvalsh(x)


def matrix_rank(vector: Tensor,
                rtol: Optional[Union[float, Tuple[float]]] = None)\
        -> Tensor:
    if rtol is None:
        return tf.linalg.matrix_rank(vector)
    if tf.size(vector) == 0:
        return 0
    if tf.size(vector) == 1:
        return tf.math.count_nonzero(vector)
    vector = tf.reshape(vector,[-1])
    vector = tf.expand_dims(vector,0)
    if hasattr(rtol,'dtype'):
        if rtol.dtype != vector.dtype:
            promoted_dtype = tf.experimental.numpy.promote_types(rtol.dtype,vector.dtype)
            vector = tf.cast(vector,promoted_dtype)
            rtol = tf.cast(rtol,promoted_dtype)
    return tf.linalg.matrix_rank(vector,rtol)

    
def cross (x1: tf.Tensor,
           x2: tf.Tensor,
           axis:int = -1) -> tf.Tensor:
    return tf.experimental.numpy.cross(x1, x2,axis=axis)


# Extra #
# ------#

def vector_to_skew_symmetric_matrix(vector: Tensor)\
        -> Tensor:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = tf.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = tf.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = tf.concat((zs, -a3s, a2s), -1)
    row2 = tf.concat((a3s, zs, -a1s), -1)
    row3 = tf.concat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return tf.concat((row1, row2, row3), -2)
