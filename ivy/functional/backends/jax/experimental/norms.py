import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"0.3.14 and below": ("float16",)}, backend_version)
def l2_normalize(x: JaxArray, /, *, axis: int = None, out=None) -> JaxArray:
    denorm = jnp.linalg.norm(x, axis=axis, ord=2, keepdims=True)
    denorm = jnp.maximum(denorm, 1e-12)
    return x / denorm
