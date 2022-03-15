# global
import jax.numpy as jnp
from typing import Tuple, Union


def min(x: jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> jnp.ndarray:
    return jnp.min(a = jnp.asarray(x), axis = axis, keepdims = keepdims)

def max(x: jnp.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> jnp.ndarray:
    return jnp.max(a = jnp.asarray(x), axis = axis, keepdims = keepdims)
