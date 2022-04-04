import jax.numpy as jnp
from typing import Tuple

from ivy.functional.backends.jax import JaxArray

def unique_all(x : JaxArray)\
                -> Tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
    return jnp.asarray(np.unique_all(np.asarray(x)))