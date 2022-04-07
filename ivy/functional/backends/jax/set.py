# global
import jax.numpy as jnp
from typing import Tuple


# local
from ivy.functional.backends.jax import JaxArray


def unique_all(x : JaxArray)\
                -> Tuple[JaxArray, JaxArray, JaxArray, JaxArray]:

    return jnp.unique_all(jnp.asarray(x))