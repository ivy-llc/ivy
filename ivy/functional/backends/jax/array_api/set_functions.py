# global
import jax.numpy as jnp
from typing import Tuple
from collections import namedtuple

def unique_counts(x: jnp.array) \
                -> Tuple[jnp.array, jnp.array]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, c = jnp.unique(x, return_counts=True)
    return uc(v, c)