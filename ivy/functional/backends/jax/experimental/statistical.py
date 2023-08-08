import jax.numpy as jnp
from typing import Optional, Union, Tuple, Sequence

from ivy.functional.backends.jax import JaxArray
import jax.lax as jlax
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ..statistical import _infer_dtype


@with_unsupported_dtypes(
    {"0.4.14 and below": ("bfloat16",)},
    backend_version,
)
def histogram(
    a: jnp.ndarray,
    /,
    *,
    bins: Optional[Union[int, jnp.ndarray]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[jnp.ndarray] = None,
    density: Optional[bool] = False,
    out: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray]:
    min_a = jnp.min(a)
    max_a = jnp.max(a)
    if isinstance(bins, jnp.ndarray) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        bins = jnp.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = jnp.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    if bins.size < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    bins_out = bins.copy()
    if extend_lower_interval and min_a < bins[0]:
        bins = bins.at[0].set(min_a)
    if extend_upper_interval and max_a > bins[-1]:
        bins = bins.at[-1].set(max_a)
    if a.ndim > 0 and axis is not None:
        inverted_shape_dims = list(jnp.flip(jnp.arange(a.ndim)))
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
                ret_1D = jnp.histogram(
                    a_1d,
                    bins=bins,
                    range=range,
                )[0]
                ret.append(ret_1D)
        else:
            weights_along_axis_1d = (
                weights.transpose(inverted_shape_dims)
                .flatten()
                .reshape((-1, shape_axes))
            )
            ret = []
            for a_1d, weights_1d in zip(a_along_axis_1d, weights_along_axis_1d):
                ret_1D = jnp.histogram(
                    a_1d,
                    weights=weights_1d,
                    bins=bins,
                    range=range,
                )[0]
                ret.append(ret_1D)
        out_shape = list(a.shape)
        for dimension in sorted(axis, reverse=True):
            del out_shape[dimension]
        out_shape.insert(0, len(bins) - 1)
        ret = jnp.array(ret)
        ret = ret.flatten()
        index = jnp.zeros(len(out_shape), dtype=int)
        ret_shaped = jnp.zeros(out_shape)
        dim = 0
        i = 0
        if list(index) == list(jnp.array(out_shape) - 1):
            ret_shaped = ret_shaped.at[tuple(index)].set(ret[i])
        while list(index) != list(jnp.array(out_shape) - 1):
            ret_shaped = ret_shaped.at[tuple(index)].set(ret[i])
            dim_full_flag = False
            while index[dim] == out_shape[dim] - 1:
                index = index.at[dim].set(0)
                dim += 1
                dim_full_flag = True
            index = index.at[dim].add(1)
            i += 1
            if dim_full_flag:
                dim = 0
        if list(index) == list(jnp.array(out_shape) - 1):
            ret_shaped = ret_shaped.at[tuple(index)].set(ret[i])
        ret = ret_shaped
    else:
        ret = jnp.histogram(
            a=a, bins=bins, range=range, weights=weights, density=density
        )[0]
    if dtype:
        ret = ret.astype(dtype)
        bins_out = jnp.array(bins_out).astype(dtype)
    # TODO: weird error when returning bins: return ret, bins_out
    return ret


@with_unsupported_dtypes(
    {"0.4.14 and below": ("complex64", "complex128")}, backend_version
)
def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    ret = jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )
    if input.dtype in [jnp.uint64, jnp.int64, jnp.float64]:
        return ret.astype(jnp.float64)
    elif input.dtype in [jnp.float16, jnp.bfloat16]:
        return ret.astype(input.dtype)
    else:
        return ret.astype(jnp.float32)


# Jax doesn't support overwrite_input=True and out!=None
def nanmean(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def quantile(
    a: JaxArray,
    q: Union[float, JaxArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    interpolation = "nearest" if interpolation == "nearest_jax" else interpolation
    return jnp.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    )


def corrcoef(
    x: JaxArray,
    /,
    *,
    y: Optional[JaxArray] = None,
    rowvar: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)


def nanmedian(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)

    if overwrite_input:
        copied_input = input.copy()
        overwrite_input = False
        out = None
        return jnp.nanmedian(
            copied_input,
            axis=axis,
            keepdims=keepdims,
            overwrite_input=overwrite_input,
            out=out,
        )

    return jnp.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=False, out=None
    )


def bincount(
    x: JaxArray,
    /,
    *,
    weights: Optional[JaxArray] = None,
    minlength: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if weights is not None:
        ret = jnp.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = jnp.bincount(x, minlength=minlength).astype(x.dtype)
    return ret


def cov(
    x1: JaxArray,
    x2: JaxArray = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[JaxArray] = None,
    aweights: Optional[JaxArray] = None,
    dtype: Optional[jnp.dtype] = None,
) -> JaxArray:
    if not dtype:
        x1 = jnp.asarray(x1, dtype=jnp.float64)

    if jnp.ndim(x1) > 2:
        raise ValueError("x1 has more than 2 dimensions")

    if x2 is not None:
        if jnp.ndim(x2) > 2:
            raise ValueError("x2 has more than 2 dimensions")

    if fweights is not None:
        fweights = jnp.asarray(fweights, dtype=jnp.int64)

    return jnp.cov(
        m=x1,
        y=x2,
        rowvar=rowVar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
    )


def cummax(
    x: JaxArray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> Tuple[JaxArray, JaxArray]:
    if x.dtype in (jnp.bool_, jnp.float16):
        x = x.astype(jnp.float64)
    elif x.dtype in (jnp.int16, jnp.int8, jnp.uint8):
        x = x.astype(jnp.int64)
    elif x.dtype in (jnp.complex128, jnp.complex64):
        x = jnp.real(x).astype(jnp.float64)

    if exclusive or (reverse and exclusive):
        if exclusive and reverse:
            indices = __find_cummax_indices(jnp.flip(x, axis=axis), axis=axis)
            x = jlax.cummax(jnp.flip(x, axis=axis), axis=axis)
            x, indices = jnp.swapaxes(x, axis, -1), jnp.swapaxes(indices, axis, -1)
            x, indices = jnp.concatenate(
                (jnp.zeros_like(x[..., -1:]), x[..., :-1]), -1
            ), jnp.concatenate(
                (jnp.zeros_like(indices[..., -1:]), indices[..., :-1]), -1
            )
            x, indices = jnp.swapaxes(x, axis, -1), jnp.swapaxes(indices, axis, -1)
            res, indices = jnp.flip(x, axis=axis), jnp.flip(indices, axis=axis)
        elif exclusive:
            x = jnp.swapaxes(x, axis, -1)
            x = jnp.concatenate((jnp.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = jnp.swapaxes(x, axis, -1)
            indices = __find_cummax_indices(x, axis=axis)
            res = jlax.cummax(x, axis=axis)
        return res, indices

    if reverse:
        y = jnp.flip(x, axis=axis)
        indices = __find_cummax_indices(y, axis=axis)
        indices = jnp.flip(indices, axis=axis)
    else:
        indices = __find_cummax_indices(x, axis=axis)
    return jlax.cummax(x, axis, reverse=reverse), indices


def __find_cummax_indices(
    x: JaxArray,
    axis: int = 0,
) -> JaxArray:
    n, indice, indices = 0, [], []

    if isinstance(x[0], JaxArray) and len(x[0].shape) >= 1:
        if axis >= 1:
            for ret1 in x:
                indice = __find_cummax_indices(ret1, axis=axis - 1)
                indices.append(indice)
        else:
            z_list = __get_index(x.tolist())
            indices, n1 = x.copy(), {}
            indices = jnp.zeros(jnp.asarray(indices.shape), dtype=x.dtype)
            z_list = sorted(z_list, key=lambda i: i[1])
            for y, y_index in z_list:
                multi_index = y_index
                if tuple(multi_index[1:]) not in n1:
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices = indices.at[y_index].set(multi_index[0])
                elif (
                    y >= x[tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))]
                ):
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices = indices.at[y_index].set(multi_index[0])
                else:
                    indices = indices.at[y_index].set(n1[tuple(multi_index[1:])])
    else:
        n, indices = 0, []
        for idx, y in enumerate(x):
            if idx == 0 or x[n] <= y:
                n = idx
            indices.append(n)

    return jnp.asarray(indices, dtype="int64")


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


@with_unsupported_dtypes({"0.4.14 and below": "bfloat16"}, backend_version)
def cummin(
    x: JaxArray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis < 0:
        axis = axis + len(x.shape)
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is jnp.bool_:
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    return jlax.cummin(x, axis, reverse=reverse).astype(dtype)


def igamma(
    a: JaxArray,
    /,
    *,
    x: JaxArray,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jlax.igamma(a=a, x=x)
