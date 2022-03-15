# global
import jax
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
from ivy import inf
from ivy.functional.backends.jax import JaxArray
import ivy as _ivy


# noinspection PyUnusedLocal,PyShadowingBuiltins
def vector_norm(x: JaxArray,
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, -inf]] = 2)\
        -> JaxArray:

    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), ord, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, ord, axis, keepdims)

    if jnp_normalized_vector.shape == ():
        return jnp.expand_dims(jnp_normalized_vector, 0)
    return jnp_normalized_vector


def svd(x:JaxArray,full_matrices: bool = True) -> Union[JaxArray, Tuple[JaxArray,...]]:
    results=namedtuple("svd", "U S Vh")
    U, D, VT=jnp.linalg.svd(x, full_matrices=full_matrices)
    res=results(U, D, VT)
    return res


def diagonal(x: JaxArray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> JaxArray:
    return jnp.diagonal(x, offset, axis1, axis2)


def qr(x: JaxArray,
       mode: str = 'reduced') -> namedtuple('qr', ['Q', 'R']):
    res = namedtuple('qr', ['Q', 'R'])
    q, r = jnp.linalg.qr(x, mode=mode)
    return res(q, r)


def matmul(x1: JaxArray,
           x2: JaxArray) -> JaxArray:
    return jnp.matmul(x1, x2)


def slogdet(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = jnp.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res


def trace(x: JaxArray,
          offset: int = 0)\
              -> JaxArray:
    return jax.numpy.trace(x, offset)
