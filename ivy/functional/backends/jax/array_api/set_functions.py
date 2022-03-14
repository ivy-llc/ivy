# global
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray
from typing import Tuple
from collections import namedtuple

def unique_counts(x: JaxArray) \
                -> Tuple[JaxArray, JaxArray]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, c = jnp.unique(x, return_counts=True)
    return uc(v, c)