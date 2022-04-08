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
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        values = jnp.append(values, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    inverse_indices = jnp.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: JaxArray) \
        -> JaxArray:
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if (nan_count > 1):
        unique = jnp.append(jnp.unique(x.flatten()), jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    else:
        unique = jnp.unique(x.flatten()).astype(x.dtype)
    return unique


def unique_counts(x: JaxArray) \
        -> Tuple[JaxArray, JaxArray]:
    v, c = jnp.unique(x, return_counts=True)
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        nan_idx = jnp.where(jnp.isnan(v))
        c = c.at[nan_idx].set(1)
        v = jnp.append(v, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
        c = jnp.append(c, jnp.full(nan_count - 1, 1)).astype('int32')
    uc = namedtuple('uc', ['values', 'counts'])
    return uc(v, c)
