# global
import jax.numpy as jnp
from typing import Tuple
from collections import namedtuple

# local
from ivy.functional.backends.jax import JaxArray


def unique_inverse(x: JaxArray) \
        -> Tuple[JaxArray, JaxArray]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    return out(values, inverse_indices)
