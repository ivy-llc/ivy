# global
import jax.numpy as _jnp
from typing import Tuple, Union


def min(x: _jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> _jnp.ndarray:
    return _jnp.min(a = _jnp.asarray(x), axis = axis, keepdims = keepdims)

def prod(x: _jnp.ndarray,
         axis: Union[int, Tuple[int]] = None,
         dtype: _jnp.dtype = None,
         keepdims: bool = False)\
        -> _jnp.ndarray:
    return _jnp.prod(x,axis,dtype,keepdims)