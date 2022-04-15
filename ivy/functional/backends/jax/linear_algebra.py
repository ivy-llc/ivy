# global
import jax
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Literal, List
from collections import namedtuple

# local
from ivy import inf
from ivy.functional.backends.jax import JaxArray
import ivy


# Array API Standard #
# -------------------#

def eigh(x: JaxArray,
         out: Optional[JaxArray] = None)\
  ->JaxArray:
         ret = jnp.linalg.eigh(x)


def pinv(x: JaxArray,
         rtol: Optional[Union[float, Tuple[float]]] = None,
         out: Optional[JaxArray] = None) \
        -> JaxArray:

    if rtol is None:
        ret = jnp.linalg.pinv(x)
    else:
        ret = jnp.linalg.pinv(x, rtol)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def matrix_transpose(x: JaxArray,
                     out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.swapaxes(x, -1, -2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# noinspection PyUnusedLocal,PyShadowingBuiltins
def vector_norm(x: JaxArray,
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, -inf]] = 2,
                out: Optional[JaxArray] = None)\
        -> JaxArray:

    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), ord, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, ord, axis, keepdims)

    if jnp_normalized_vector.shape == ():
        ret = jnp.expand_dims(jnp_normalized_vector, 0)
    else:
        ret = jnp_normalized_vector
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret



def matrix_norm(x: JaxArray,
                ord: Optional[Union[int, float, Literal[inf, - inf, 'fro', 'nuc']]] = 'fro',
                keepdims: bool = False,
                out: Optional[JaxArray] = None)\
        -> JaxArray:
    if x.size == 0:
        if keepdims:
            ret = x.reshape(x.shape[:-2] + (1, 1))
        else:
            ret = x.reshape(x.shape[:-2])
    else:
        ret = jnp.linalg.norm(x, ord, (-2, -1), keepdims)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret



def svd(x: JaxArray,
        full_matrices: bool = True,
        out: Optional[JaxArray] = None)\
        -> Union[JaxArray, Tuple[JaxArray,...]]:
    results = namedtuple("svd", "U S Vh")
    U, D, VT = jnp.linalg.svd(x, full_matrices=full_matrices)
    ret = results(U, D, VT)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def outer(x1: JaxArray,
          x2: JaxArray,
          out: Optional[JaxArray] = None)\
        -> JaxArray:
    return jnp.outer(x1, x2, out=out)


def diagonal(x: JaxArray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1,
             out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.diagonal(x, offset, axis1, axis2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def svdvals(x: JaxArray,
            out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.linalg.svd(x, compute_uv=False)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def qr(x: JaxArray,
       mode: str = 'reduced',
       out: Optional[JaxArray] = None) \
        -> namedtuple('qr', ['Q', 'R']):
    res = namedtuple('qr', ['Q', 'R'])
    q, r = jnp.linalg.qr(x, mode=mode)
    ret = res(q, r)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def matmul(x1: JaxArray,
           x2: JaxArray,
           out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.matmul(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def slogdet(x:Union[ivy.Array,ivy.NativeArray],
            out: Optional[JaxArray] = None)\
        -> Union[ivy.Array, Tuple[ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = jnp.linalg.slogdet(x)
    ret = results(sign, logabsdet)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tensordot(x1: JaxArray, x2: JaxArray,
              axes: Union[int, Tuple[List[int], List[int]]] = 2,
              out: Optional[JaxArray] = None) \
        -> JaxArray:

    ret = jnp.tensordot(x1, x2, axes)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def trace(x: JaxArray,
          offset: int = 0,
          out: Optional[JaxArray] = None)\
              -> JaxArray:
    return jax.numpy.trace(x, offset, out=out)


def det(x:jnp.array,
        out: Optional[JaxArray] = None) \
    -> jnp.array:
    ret = jnp.linalg.det(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def cholesky(x: JaxArray, 
             upper: bool = False,
             out: Optional[JaxArray] = None) -> JaxArray:
    if not upper:
        ret = jnp.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = jnp.transpose(jnp.linalg.cholesky(jnp.transpose(x, axes=axes)),
                        axes=axes)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret



def eigvalsh(x: JaxArray,
             out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.linalg.eigvalsh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def inv(x: JaxArray,
        out: Optional[JaxArray] = None)\
        -> JaxArray:
    if jnp.any(jnp.linalg.det(x.astype('float64')) == 0):
        ret = x
    else:
        ret = jnp.linalg.inv(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def matrix_rank(vector: JaxArray,
                rtol: Optional[Union[float, Tuple[float]]] = None,
                out: Optional[JaxArray] = None) \
        -> JaxArray:
        if vector.size == 0:
            ret = 0
        elif vector.size == 1:
            ret = jnp.count_nonzero(vector)
        else:
            if vector.ndim >2:
                vector = vector.reshape([-1])
            ret = jnp.linalg.matrix_rank(vector, rtol)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret


def cross (x1: JaxArray,
           x2: JaxArray,
           axis:int = -1,
           out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.cross(a= x1, b = x2, axis= axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def vecdot(x1: JaxArray,
           x2: JaxArray,
           axis: int = -1,
           out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.tensordot(x1, x2, (axis, axis))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret

# Extra #
# ------#

def vector_to_skew_symmetric_matrix(vector: JaxArray,
                                    out: Optional[JaxArray] = None)\
        -> JaxArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = jnp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = jnp.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = jnp.concatenate((zs, -a3s, a2s), -1)
    row2 = jnp.concatenate((a3s, zs, -a1s), -1)
    row3 = jnp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    ret = jnp.concatenate((row1, row2, row3), -2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
