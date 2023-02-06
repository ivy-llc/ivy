from typing import Optional, Union, Tuple, Sequence

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

import numpy as np


# TODO: Implement bins as str
#       Out does not work.
@with_unsupported_dtypes(
    {
        "1.23.0 and below": (
            "bfloat16",
            "float16",
        )
    },
    backend_version,
)
def histogram(
    a: np.ndarray,
    /,
    *,
    bins: Optional[Union[int, Sequence[int], str]] = None,
    axis: Optional[np.ndarray] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[np.ndarray] = None,
    density: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    if range:
        bins = np.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    bins_out = bins.copy()
    if extend_lower_interval:
        bins[0] = -np.inf
    if extend_upper_interval:
        bins[-1] = np.inf
    if axis is not None:
        histogram_values = np.apply_along_axis(
            lambda x: np.histogram(
                a=x,
                bins=bins,
                range=range,
                weights=weights,
            )[0],
            axis,
            a,
        )
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = np.array(bins_out).astype(dtype)
        return histogram_values, bins_out
    else:
        ret = np.histogram(
            a=a, bins=bins, range=range, weights=weights, density=density
        )
        histogram_values = ret[0]
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = np.array(bins_out).astype(dtype)
        return histogram_values, bins_out


def median(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


def nanmean(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


nanmean_support_native_out = True


def unravel_index(
    indices: np.ndarray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.unravel_index(indices, shape)


def quantile(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # quantile method in numpy backend, always return an array with dtype=float64.
    # in other backends, the output is the same dtype as the input.

    tuple(axis) if isinstance(axis, list) else axis

    return np.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    ).astype(a.dtype)


def corrcoef(
    x: np.ndarray,
    /,
    *,
    y: Optional[np.ndarray] = None,
    rowvar: Optional[bool] = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)
