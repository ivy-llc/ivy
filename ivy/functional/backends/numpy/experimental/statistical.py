from typing import Optional, Union, Tuple, Sequence
import numpy as np

import ivy  # noqa
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {"1.25.0 and below": ("bfloat16",)},
    backend_version,
)
def histogram(
    a: np.ndarray,
    /,
    *,
    bins: Optional[Union[int, np.ndarray]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[np.ndarray] = None,
    density: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    min_a = np.min(a)
    max_a = np.max(a)
    if isinstance(bins, np.ndarray) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        bins = np.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = np.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    if bins.size < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    bins_out = bins.copy()
    if extend_lower_interval and min_a < bins[0]:
        bins[0] = min_a
    if extend_upper_interval and max_a > bins[-1]:
        bins[-1] = max_a
    if a.ndim > 0 and axis is not None:
        inverted_shape_dims = list(np.flip(np.arange(a.ndim)))
        if isinstance(axis, int):
            axis = [axis]
        shape_axes = 1
        for dimension in axis:
            inverted_shape_dims.remove(dimension)
            inverted_shape_dims.append(dimension)
            shape_axes *= a.shape[dimension]
        a_along_axis_1d = (
            a.transpose(inverted_shape_dims).flatten().reshape((-1, shape_axes))
        )
        if weights is None:
            ret = []
            for a_1d in a_along_axis_1d:
                ret_1d = np.histogram(
                    a_1d,
                    bins=bins,
                    range=range,
                    # TODO: waiting tensorflow version support to density
                    # density=density,
                )[0]
                ret.append(ret_1d)
        else:
            weights_along_axis_1d = (
                weights.transpose(inverted_shape_dims)
                .flatten()
                .reshape((-1, shape_axes))
            )
            ret = []
            for a_1d, weights_1d in zip(a_along_axis_1d, weights_along_axis_1d):
                ret_1d = np.histogram(
                    a_1d,
                    weights=weights_1d,
                    bins=bins,
                    range=range,
                    # TODO: waiting tensorflow version support to density
                    # density=density,
                )[0]
                ret.append(ret_1d)
        out_shape = list(a.shape)
        for dimension in sorted(axis, reverse=True):
            del out_shape[dimension]
        out_shape.insert(0, len(bins) - 1)
        ret = np.array(ret)
        ret = ret.flatten()
        index = np.zeros(len(out_shape), dtype=int)
        ret_shaped = np.zeros(out_shape)
        dim = 0
        i = 0
        if list(index) == list(np.array(out_shape) - 1):
            ret_shaped[tuple(index)] = ret[i]
        while list(index) != list(np.array(out_shape) - 1):
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
        if list(index) == list(np.array(out_shape) - 1):
            ret_shaped[tuple(index)] = ret[i]
        ret = ret_shaped
    else:
        ret = np.histogram(
            a=a, bins=bins, range=range, weights=weights, density=density
        )[0]
    if dtype:
        ret = ret.astype(dtype)
        bins_out = np.array(bins_out).astype(dtype)
    # TODO: weird error when returning bins: return ret, bins_out
    return ret


def median(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is not None:
        out = np.reshape(out, input.shape)
    ret = np.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )
    if input.dtype in [np.uint64, np.int64, np.float64]:
        return ret.astype(np.float64)
    elif input.dtype in [np.float16]:
        return ret.astype(input.dtype)
    else:
        return ret.astype(np.float32)


median.support_native_out = True


def nanmean(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


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

    (tuple(axis) if isinstance(axis, list) else axis)

    return np.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    ).astype(a.dtype)


def corrcoef(
    x: np.ndarray,
    /,
    *,
    y: Optional[np.ndarray] = None,
    rowvar: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)


def nanmedian(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def bincount(
    x: np.ndarray,
    /,
    *,
    weights: Optional[np.ndarray] = None,
    minlength: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is not None:
        ret = np.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = np.bincount(x, minlength=minlength)
        ret = ret.astype(x.dtype)
    return ret


bincount.support_native_out = False
