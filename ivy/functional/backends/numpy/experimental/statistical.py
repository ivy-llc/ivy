from typing import Optional, Union, Tuple, Sequence

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

import numpy as np

import ivy  # noqa
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


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
    bins: Optional[Union[int, np.ndarray, str]] = None,
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
    if a.ndim > 0 and axis is not None:
        inverted_shape_dims = list(np.flip(np.arange(a.ndim)))
        inverted_shape_dims.remove(axis)
        inverted_shape_dims.append(axis)
        a_along_axis_1d = a.transpose(inverted_shape_dims).flatten().reshape((-1, a.shape[axis]))
        if weights is None:
            ret = []
            for a_1d in a_along_axis_1d:
                ret_1D = np.histogram(
                    a_1d,
                    bins=bins,
                    range=range,
                )[0]
                ret.append(ret_1D)
        else:
            weights_along_axis_1d = weights.transpose(inverted_shape_dims).flatten().reshape((-1, weights.shape[axis]))
            ret = []
            for a_1d, weights_1d in zip(a_along_axis_1d, weights_along_axis_1d):
                ret_1D = np.histogram(
                    a_1d,
                    weights=weights_1d,
                    bins=bins,
                    range=range,
                )[0]
                ret.append(ret_1D)
        out_shape = list(a.shape)
        del out_shape[axis]
        out_shape.insert(0, len(bins)-1)
        ret = np.array(ret)
        ret = ret.flatten()
        index = np.zeros(len(out_shape), dtype=int)
        ret_shaped = np.zeros(out_shape)
        dim = 0
        i = 0
        if list(index) == list(np.array(out_shape)-1):
            ret_shaped[tuple(index)] = ret[i]
        while list(index) != list(np.array(out_shape)-1):
            ret_shaped[tuple(index)] = ret[i]
            dim_full_flag = False
            while index[dim] == out_shape[dim] - 1:
                index[dim] = 0
                dim += 1
                dim_full_flag = True
            index[dim] += 1
            i += 1
            if dim_full_flag:
                dim = 0
        if list(index) == list(np.array(out_shape)-1):
            ret_shaped[tuple(index)] = ret[i]
        ret = ret_shaped
    else:
        ret = np.histogram(
            a=a, bins=bins, range=range, weights=weights, density=density
        )[0]
    if dtype:
        ret = ret.astype(dtype)
        bins_out = np.array(bins_out).astype(dtype)
    return ret, bins_out


def median(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is not None:
        out = np.reshape(out, input.shape)
    return np.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


median.support_native_out = True


def nanmean(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_supported_dtypes({"1.23.0 and below": ("int32", "int64")}, backend_version)
def unravel_index(
    indices: np.ndarray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple:
    ret = np.asarray(np.unravel_index(indices, shape), dtype=np.int32)
    return tuple(ret)


unravel_index.support_native_out = False


def quantile(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
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


def nanmedian(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True
