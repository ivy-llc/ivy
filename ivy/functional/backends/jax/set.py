# global
import jax.numpy as jnp
from typing import NamedTuple, Optional
from collections import namedtuple

# local
from ivy.functional.backends.jax import JaxArray


def unique_all(
    x: JaxArray,
    /,
) -> NamedTuple:
    UniqueAll = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
    )

    values, indices, inverse_indices, counts = jnp.unique(
        x, return_index=True, return_counts=True, return_inverse=True
    )
    nan_count = jnp.sum(jnp.isnan(x)).item()

    if nan_count > 1:
        values = jnp.concatenate(
            (
                values,
                jnp.full(
                    fill_value=jnp.nan, shape=(nan_count - 1,), dtype=values.dtype
                ),
            ),
            axis=0,
        )
        counts = jnp.concatenate(
            (
                counts[:-1],
                jnp.full(fill_value=1, shape=(nan_count,), dtype=counts.dtype),
            ),
            axis=0,
        )

        nan_idx = jnp.where(jnp.isnan(x.flatten()))[0]

        indices = jnp.concatenate((indices[:-1], nan_idx), axis=0).astype(indices.dtype)
    else:
        pass

    return UniqueAll(
        values.astype(x.dtype), indices, jnp.reshape(inverse_indices, x.shape), counts
    )


def unique_counts(
    x: JaxArray,
    /,
) -> NamedTuple:
    v, c = jnp.unique(x, return_counts=True)
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        nan_idx = jnp.where(jnp.isnan(v))
        c = c.at[nan_idx].set(1)
        v = jnp.append(v, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
        c = jnp.append(c, jnp.full(nan_count - 1, 1)).astype("int32")
    uc = namedtuple("uc", ["values", "counts"])
    return uc(v, c)


def unique_inverse(
    x: JaxArray,
    /,
) -> NamedTuple:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        values = jnp.append(values, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    inverse_indices = jnp.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        unique = jnp.append(
            jnp.unique(x.flatten()), jnp.full(nan_count - 1, jnp.nan)
        ).astype(x.dtype)
    else:
        unique = jnp.unique(x.flatten()).astype(x.dtype)
    return unique
