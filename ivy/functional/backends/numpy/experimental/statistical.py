from typing import Optional, Union, Tuple, Sequence
import numpy as np
import math
import ivy  # noqa
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ..statistical import _infer_dtype
from copy import deepcopy


@with_unsupported_dtypes(
    {"1.25.1 and below": ("bfloat16",)},
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


def _validate_quantile(q):
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (np.all(0 <= q) and np.all(q <= 1)):
            return False
    return True


def _to_positive_axis(axis, ndim):
    if not isinstance(axis, (list, tuple)):
        axis = [axis]

    if len(axis) == 0:
        raise ValueError("Axis can't be empty!")

    if len(set(axis)) != len(axis):
        raise ValueError("Duplicated axis!")

    for i in range(len(axis)):
        if not (isinstance(axis[i], int) and (ndim > axis[i] >= -ndim)):
            raise ValueError("Axis must be int in range [-rank(x), rank(x))")
        if axis[i] < 0:
            axis[i] += ndim
    return axis


def _handle_axis(a, q, fn, keepdims=False, axis=None):
    nd = a.ndim
    axis_arg = deepcopy(axis)
    if axis is not None:
        axis = _to_positive_axis(axis, nd)

        if len(axis) == 1:
            axis_arg = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)

            for i, s in enumerate(sorted(keep)):
                a = np.moveaxis(a, s, i)
            a = a.reshape(a.shape[:nkeep] + (-1,))
            axis_arg = -1

    ret = fn(a, q, axis=axis_arg)

    if keepdims:
        if axis is None:
            index_ret = (None,) * nd
        else:
            index_ret = tuple(None if i in axis else slice(None) for i in range(nd))
        ret = ret[(Ellipsis,) + index_ret]

    return ret


def _quantile(a, q, axis=None):
    ret_dtype = a.dtype
    if q.ndim > 2:
        raise ValueError("q argument must be a scalar or 1-dimensional!")
    if axis is None:
        axis = 0
        a = a.flatten()

    n = a.shape[axis]
    if axis != 0:
        a = np.moveaxis(a, axis, 0)

    indices = []
    for q_num in q:
        index = q_num * (n - 1)
        indices.append(index)

    a.sort(0)
    outputs = []

    for index in indices:
        indices_below = np.floor(index).astype(np.int32)
        indices_upper = np.ceil(index).astype(np.int32)

        weights = index - indices_below.astype("float64")

        indices_below = np.clip(indices_below, 0, n - 1)
        indices_upper = np.clip(indices_upper, 0, n - 1)
        tensor_upper = np.take(a, indices_upper, axis=0)  # , mode="clip")
        tensor_below = np.take(a, indices_below, axis=0)  # , mode="clip")

        pred = weights <= 0.5
        out = np.where(pred, tensor_below, tensor_upper)
        outputs.append(out)
    return np.array(outputs, dtype=ret_dtype)


def _compute_quantile_wrapper(
    x, q, axis=None, keepdims=False, interpolation="linear", out=None, nearest_jax=True
):
    if not _validate_quantile(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if interpolation in ["linear", "lower", "higher", "midpoint", "nearest"]:
        if interpolation == "nearest" and nearest_jax:
            return _handle_axis(x, q, _quantile, keepdims=keepdims, axis=axis)
        else:
            axis = tuple(axis) if isinstance(axis, list) else axis

            return np.quantile(
                x, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
            ).astype(x.dtype)
    else:
        raise ValueError(
            "Interpolation must be 'linear', 'lower', 'higher', 'midpoint' or 'nearest'"
        )


def quantile(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[np.ndarray] = None,
    nearest_jax: Optional[bool] = True,
) -> np.ndarray:
    # quantile method in numpy backend, always return an array with dtype=float64.
    # in other backends, the output is the same dtype as the input.
    # added the nearest_jax mode to enable jax-like calculations for method="nearest"
    return _compute_quantile_wrapper(
        a,
        q,
        axis=axis,
        keepdims=keepdims,
        interpolation=interpolation,
        out=out,
        nearest_jax=nearest_jax,
    )


def corrcoef(
    x: np.ndarray,
    /,
    *,
    y: Optional[np.ndarray] = None,
    rowvar: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)


@with_unsupported_dtypes(
    {"1.25.0 and below": ("bfloat16",)},
    backend_version,
)
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


def cov(
    x1: np.ndarray,
    x2: np.ndarray = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[np.ndarray] = None,
    aweights: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    if fweights is not None:
        fweights = fweights.astype(np.int64)

    return np.cov(
        m=x1,
        y=x2,
        rowvar=rowVar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        dtype=dtype,
    )


cov.support_native_out = False


def cummax(
    x: np.ndarray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if x.dtype in (np.bool_, np.float16):
        x = x.astype(np.float64)
    elif x.dtype in (np.int16, np.int8, np.uint8):
        x = x.astype(np.int64)
    elif x.dtype in (np.complex128, np.complex64):
        x = np.real(x).astype(np.float64)

    if exclusive or reverse:
        if exclusive and reverse:
            indices = __find_cummax_indices(np.flip(x, axis=axis), axis=axis)
            x = np.maximum.accumulate(np.flip(x, axis=axis), axis=axis, dtype=x.dtype)
            x = np.swapaxes(x, axis, -1)
            indices = np.swapaxes(indices, axis, -1)
            x, indices = np.concatenate(
                (np.zeros_like(x[..., -1:]), x[..., :-1]), -1
            ), np.concatenate((np.zeros_like(indices[..., -1:]), indices[..., :-1]), -1)
            x, indices = np.swapaxes(x, axis, -1), np.swapaxes(indices, axis, -1)
            res, indices = np.flip(x, axis=axis), np.flip(indices, axis=axis)

        elif exclusive:
            x = np.swapaxes(x, axis, -1)
            x = np.concatenate((np.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = np.swapaxes(x, axis, -1)
            indices = __find_cummax_indices(x, axis=axis)
            res = np.maximum.accumulate(x, axis=axis, dtype=x.dtype)
        elif reverse:
            x = np.flip(x, axis=axis)
            indices = __find_cummax_indices(x, axis=axis)
            x = np.maximum.accumulate(x, axis=axis)
            res, indices = np.flip(x, axis=axis), np.flip(indices, axis=axis)
        return res, indices
    indices = __find_cummax_indices(x, axis=axis)
    return np.maximum.accumulate(x, axis=axis, dtype=x.dtype), indices


def __find_cummax_indices(
    x: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    indices = []
    if type(x[0]) == np.ndarray:
        if axis >= 1:
            for ret1 in x:
                indice = __find_cummax_indices(ret1, axis=axis - 1)
                indices.append(indice)

        else:
            indice_list = __get_index(x.tolist())
            indices, n1 = x.copy(), {}
            indices.fill(0)
            indice_list = sorted(indice_list, key=lambda i: i[1])
            for y, y_index in indice_list:
                multi_index = y_index
                if tuple(multi_index[1:]) not in n1:
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                elif (
                    y >= x[tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))]
                ):
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                else:
                    indices[y_index] = n1[tuple(multi_index[1:])]
    else:
        n = 0
        for index1, ret1 in enumerate(x):
            if x[n] <= ret1 or index1 == 0:
                n = index1
            indices.append(n)
    return np.array(indices, dtype=np.int64)


def __get_index(lst, indices=None, prefix=None):
    if indices is None:
        indices = []
    if prefix is None:
        prefix = []

    if isinstance(lst, list):
        for i, sub_lst in enumerate(lst):
            sub_indices = prefix + [i]
            __get_index(sub_lst, indices, sub_indices)
    else:
        indices.append((lst, tuple(prefix)))
    return indices


@with_unsupported_dtypes({"1.25.1 and below": "bfloat16"}, backend_version)
def cummin(
    x: np.ndarray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if not (reverse):
        return np.minimum.accumulate(x, axis, dtype=dtype, out=out)
    elif reverse:
        x = np.minimum.accumulate(np.flip(x, axis=axis), axis=axis, dtype=dtype)
        return np.flip(x, axis=axis)


def igamma(
    a: np.ndarray,
    /,
    *,
    x: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    def igamma_cal(a, x):
        t = np.linspace(0, x, 10000, dtype=np.float64)
        y = np.exp(-t) * (t ** (a - 1))
        integral = np.trapz(y, t)
        return integral / math.gamma(a)

    igamma_vec = np.vectorize(igamma_cal)
    return igamma_vec(a, x).astype(a.dtype)
