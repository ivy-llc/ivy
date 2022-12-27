import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray


def logit(x: JaxArray,
          /,
          *,
          eps: Optional[float] = None,
          out=None):
    if eps is None:
        x = jnp.where(jnp.logical_or(x > 1, x < 0), jnp.nan, x)
    else:
        x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))
