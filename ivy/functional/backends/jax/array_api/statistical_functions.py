import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


def var(x: JaxArray,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> JaxArray:
    ddof = int(correction)
    return jnp.var(x, axis=axis,ddof=ddof, keepdims=keepdims)


def min(x: JaxArray,
        axis: Union[int, Tuple[int]] = None,
        keepdims = False, device = None) \
        -> jnp.ndarray:
    return jnp.min(a = jnp.asarray(x), axis = axis, keepdims = keepdims)
