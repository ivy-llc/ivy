# global
import jax.numpy as _jnp
from typing import Tuple, Union, Optional , List


def min(x: _jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> _jnp.ndarray:
    return _jnp.min(a = _jnp.asarray(x), axis = axis, keepdims = keepdims)


def prod(x: _jnp.ndarray,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[_jnp.dtype] = None,
         keepdims: bool = False)\
        -> _jnp.ndarray:
        
    if dtype == None and _jnp.issubdtype(x.dtype,_jnp.integer):
        if _jnp.issubdtype(x.dtype,_jnp.signedinteger) and x.dtype in [_jnp.int8,_jnp.int16,_jnp.int32]:
            dtype = _jnp.int32
        elif _jnp.issubdtype(x.dtype,_jnp.unsignedinteger) and x.dtype in [_jnp.uint8,_jnp.uint16,_jnp.uint32]:
            dtype = _jnp.uint32
        elif x.dtype == _jnp.int64: 
            dtype = _jnp.int64
        else:
            dtype = _jnp.uint64
            
    return _jnp.prod(a=x,axis=axis,dtype=dtype,keepdims=keepdims)