# global
import numpy as np
from typing import Tuple, Optional
from collections import namedtuple
from packaging import version

# local
import ivy


def unique_all(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    values, indices, inverse_indices, counts = np.unique(
        x,
        return_index=True,
        return_counts=True,
        return_inverse=True,
        axis=axis,
    )

    nan_count = np.sum(np.isnan(x)).item()
    if (nan_count > 1) & (np.sum(np.isnan(values)).item() == 1):
        counts[np.where(np.isnan(values))[0]] = 1
        counts = np.append(counts, np.full(fill_value=1, shape=(nan_count - 1,)))

        values = np.append(
            values, np.full(fill_value=np.nan, shape=(nan_count - 1,)), axis=0
        )
        nan_idx = np.where(np.isnan(x.flatten()))[0]
        indices = np.concatenate((indices[:-1], nan_idx), axis=0)

    if not by_value:
        sort_idx = np.argsort(indices)
        values = np.take(values, sort_idx, axis=axis)
        counts = np.take(counts, sort_idx)
        indices = np.take(indices, sort_idx)
        inv_sort_idx = ivy.current_backend().invert_permutation(sort_idx)
        inverse_indices = np.vectorize(lambda y: np.take(inv_sort_idx, y))(
            inverse_indices
        )

    return Results(
        values.astype(x.dtype),
        indices,
        inverse_indices,
        counts,
    )


def unique_counts(
    x: np.ndarray,
    /,
) -> Tuple[np.ndarray, np.ndarray]:
    v, c = np.unique(x, return_counts=True)
    nan_count = np.count_nonzero(np.isnan(x))
    if nan_count > 1:
        nan_idx = np.where(np.isnan(v))
        c[nan_idx] = 1
        v = np.append(v, np.full(nan_count - 1, np.nan)).astype(x.dtype)
        c = np.append(c, np.full(nan_count - 1, 1)).astype("int32")
    Results = namedtuple("Results", ["values", "counts"])
    return Results(v, c)


def unique_inverse(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    Results = namedtuple("Results", ["values", "inverse_indices"])
    values, inverse_indices = np.unique(x, return_inverse=True, axis=axis)
    nan_count = np.count_nonzero(np.isnan(x))
    if nan_count > 1:
        values = np.append(values, np.full(nan_count - 1, np.nan), axis=axis).astype(
            x.dtype
        )
    inverse_indices = np.reshape(inverse_indices, x.shape)
    return Results(values, inverse_indices)


def unique_values(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    nan_count = np.count_nonzero(np.isnan(x))
    if version.parse(np.__version__) >= version.parse("1.21.0") and nan_count > 1:
        unique = np.append(
            np.unique(x.flatten()), np.full(nan_count - 1, np.nan)
        ).astype(x.dtype)
    else:
        unique = np.unique(x.flatten()).astype(x.dtype)
    return unique
