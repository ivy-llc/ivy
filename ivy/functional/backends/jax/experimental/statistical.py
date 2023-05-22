from typing import Optional, Union, Tuple, Sequence

from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {"0.4.10 and below": ("bfloat16",)},
    backend_version,
)
def histogram(
    a: jnp.ndarray,
    /,
    *,
    bins: Optional[Union[int, jnp.ndarray, str]] = None,
    axis: Optional[jnp.ndarray] = None,
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
    {"0.4.10 and below": ("complex64", "complex128")}, backend_version
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
    if isinstance(axis, list):
        axis = tuple(axis)

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
