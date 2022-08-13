# global
import jax.numpy as jnp
from typing import Union, Optional, Sequence, Any

# local
from ivy.functional.backends.jax import JaxArray


def all(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.all(x, axis, keepdims=keepdims)


def any(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.any(x, axis, keepdims=keepdims)

def is_tensor(x: Any) -> bool:
    return False