# global
import jax.numpy as jnp
from typing import Tuple, Optional
from collections import namedtuple

# local
from ivy.functional.backends.jax import JaxArray
import ivy


def unique_all(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    values, indices, inverse_indices, counts = jnp.unique(
        x,
        return_index=True,
        return_counts=True,
        return_inverse=True,
        axis=axis,
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

    if not by_value:
        sort_idx = jnp.argsort(indices)
        values = jnp.take(values, sort_idx, axis=axis)
        counts = jnp.take(counts, sort_idx)
        indices = jnp.take(indices, sort_idx)
        inv_sort_idx = ivy.current_backend().invert_permutation(sort_idx)
        inverse_indices = jnp.vectorize(lambda y: jnp.take(inv_sort_idx, y))(
            inverse_indices
        )

    return Results(
        values.astype(x.dtype),
        indices,
        inverse_indices,
        counts,
    )


def unique_counts(
    x: JaxArray,
    /,
) -> Tuple[JaxArray, JaxArray]:
    v, c = jnp.unique(x, return_counts=True)
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        nan_idx = jnp.where(jnp.isnan(v))
        c = c.at[nan_idx].set(1)
        v = jnp.append(v, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
        c = jnp.append(c, jnp.full(nan_count - 1, 1)).astype("int32")
    Results = namedtuple("Results", ["values", "counts"])
    return Results(v, c)


def unique_inverse(
    x: JaxArray,
    /,
) -> Tuple[JaxArray, JaxArray]:
    Results = namedtuple("Results", ["values", "inverse_indices"])
    values, inverse_indices = jnp.unique(x, return_inverse=True)
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        values = jnp.append(values, jnp.full(nan_count - 1, jnp.nan)).astype(x.dtype)
    inverse_indices = jnp.reshape(inverse_indices, x.shape)
    return Results(values, inverse_indices)


def unique_values(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    nan_count = jnp.count_nonzero(jnp.isnan(x))
    if nan_count > 1:
        unique = jnp.append(
            jnp.unique(x.flatten()), jnp.full(nan_count - 1, jnp.nan)
        ).astype(x.dtype)
    else:
        unique = jnp.unique(x.flatten()).astype(x.dtype)
    return unique
