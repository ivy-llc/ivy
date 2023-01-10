import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray


def l2_normalize(x: JaxArray,
                 axis: int = None,
                 out=None
                 ) -> JaxArray:
    norm = jnp.linalg.norm(x, axis=axis, ord=2, keepdims=True)
    return x / norm
