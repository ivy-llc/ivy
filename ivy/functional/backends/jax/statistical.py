# global
import jax.numpy as jnp
from typing import Tuple, Union, Optional , List


# Array API Standard #
# -------------------#
import ivy


def min(x: jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> jnp.ndarray:
    return jnp.min(a = jnp.asarray(x), axis = axis, keepdims = keepdims)


def sum(x: jnp.ndarray,
        axis: Optional[Union[int,Tuple[int]]] = None,
        dtype: Optional[jnp.dtype] = None,
        keepdims: bool = False) -> jnp.ndarray:

    if dtype == None and jnp.issubdtype(x.dtype, jnp.integer):
        if jnp.issubdtype(x.dtype, jnp.signedinteger) and x.dtype in [jnp.int8, jnp.int16, jnp.int32]:
            dtype = jnp.int32
        elif jnp.issubdtype(x.dtype, jnp.unsignedinteger) and x.dtype in [jnp.uint8, jnp.uint16, jnp.uint32]:
            dtype = jnp.uint32
        elif x.dtype == jnp.int64:
            dtype = jnp.int64
        else:
            dtype = jnp.uint64

    return jnp.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def prod(x: jnp.ndarray,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[jnp.dtype] = None,
         keepdims: bool = False)\
        -> jnp.ndarray:

    if dtype == None and jnp.issubdtype(x.dtype,jnp.integer):
        if jnp.issubdtype(x.dtype,jnp.signedinteger) and x.dtype in [jnp.int8,jnp.int16,jnp.int32]:
            dtype = jnp.int32
        elif jnp.issubdtype(x.dtype,jnp.unsignedinteger) and x.dtype in [jnp.uint8,jnp.uint16,jnp.uint32]:
            dtype = jnp.uint32
        elif x.dtype == jnp.int64: 
            dtype = jnp.int64
        else:
            dtype = jnp.uint64

    return jnp.prod(a=x,axis=axis,dtype=dtype,keepdims=keepdims)


def max(x: jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> jnp.ndarray:
    return jnp.max(a = jnp.asarray(x), axis = axis, keepdims = keepdims)


def var(x: jnp.ndarray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> jnp.ndarray:
    return jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims)


# Extra #
# ------#

def einsum(equation, *operands):
    return jnp.einsum(equation, *operands)
