# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Literal

# local
from ivy import inf
from ivy.functional.backends.jax import JaxArray


def vector_norm(x: JaxArray, 
                p: Union[int, float, Literal[inf, -inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                    -> JaxArray:

    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), p, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, p, axis, keepdims)

    if jnp_normalized_vector.shape == ():
        return jnp.expand_dims(jnp_normalized_vector, 0)
    return jnp_normalized_vector
