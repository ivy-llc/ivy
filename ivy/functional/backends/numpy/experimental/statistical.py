from typing import Optional, Union, Tuple, Sequence
import numpy as np


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
) -> Tuple[np.ndarray]:
    ret = np.histogram(
        a=a,
        bins=bins,
        range=range,
        weights=weights,
        density=density
    )
    if extend_lower_interval:
        if density:
            ret[0][:] *= a[(a > range[0]) & (a < range[1])].size
        if extend_upper_interval:
            ret[0][0] += a[a < range[0]].size
            ret[0][-1] += a[a > range[1]].size
            if density:
                ret[0][:] /= a.size
        else:
            ret[0][0] += a[a < range[0]].size
            if density:
                ret[0][:] /= a[a < range[1]].size
    elif extend_upper_interval:
        if density:
            ret[0][:] *= a[(a > range[0]) & (a < range[1])].size
        ret[0][-1] += a[a > range[1]].size
        if density:
            ret[0][:] /= a[a > range[0]].size
    if dtype:
        ret[0].astype(dtype)
        ret[1].astype(dtype)
    return ret


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
