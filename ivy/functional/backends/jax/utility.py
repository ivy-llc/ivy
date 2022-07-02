# global
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


def all(
    x: JaxArray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
) -> JaxArray:
    return jnp.all(x, axis, keepdims=keepdims)


def any(
    x: JaxArray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
) -> JaxArray:
    return jnp.any(x, axis, keepdims=keepdims)
