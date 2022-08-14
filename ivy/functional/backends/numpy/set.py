# global
import numpy as np
from typing import NamedTuple, Optional
from collections import namedtuple
from packaging import version


def unique_all(x: np.ndarray) -> NamedTuple:
    UniqueAll = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
    )

    values, indices, inverse_indices, counts = np.unique(
        x, return_index=True, return_counts=True, return_inverse=True
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
    else:
        pass

    return UniqueAll(
        values.astype(x.dtype),
        indices,
        np.reshape(inverse_indices, x.shape),
        counts,
    )


def unique_counts(x: np.ndarray) -> NamedTuple:
    v, c = np.unique(x, return_counts=True)
    nan_count = np.count_nonzero(np.isnan(x))
    if nan_count > 1:
        nan_idx = np.where(np.isnan(v))
        c[nan_idx] = 1
        v = np.append(v, np.full(nan_count - 1, np.nan)).astype(x.dtype)
        c = np.append(c, np.full(nan_count - 1, 1)).astype("int32")
    uc = namedtuple("uc", ["values", "counts"])
    return uc(v, c)


def unique_inverse(x: np.ndarray) -> NamedTuple:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = np.unique(x, return_inverse=True)
    nan_count = np.count_nonzero(np.isnan(x))
    if nan_count > 1:
        values = np.append(values, np.full(nan_count - 1, np.nan)).astype(x.dtype)
    inverse_indices = inverse_indices.reshape(x.shape)
    return out(values, inverse_indices)


def unique_values(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    nan_count = np.count_nonzero(np.isnan(x))
    if version.parse(np.__version__) >= version.parse("1.21.0") and nan_count > 1:
        unique = np.append(
            np.unique(x.flatten()), np.full(nan_count - 1, np.nan)
        ).astype(x.dtype)
    else:
        unique = np.unique(x.flatten()).astype(x.dtype)
    return unique
