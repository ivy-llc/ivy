# global
import jax.numpy as _jnp
from typing import Tuple, Union


def min(x: _jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> _jnp.ndarray:
    return _jnp.min(a = _jnp.asarray(x), axis = axis, keepdims = keepdims)

def max(x: _jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> _jnp.ndarray:
    return _jnp.max(a = _jnp.asarray(x), axis = axis, keepdims = keepdims)
