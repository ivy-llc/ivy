from typing import Optional, Union, Tuple, Sequence

import numpy as np

from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
        )
    },
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
    if range:
        bins = jnp.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=a.dtype)
        range = None
    bins_out = bins.copy()
    if extend_lower_interval:
        bins = bins.at[0].set(-jnp.inf)
    if extend_upper_interval:
        bins = bins.at[-1].set(jnp.inf)
    if a.ndim > 0 and axis is not None:
        inverted_shape_dims = list(jnp.flip(jnp.arange(a.ndim)))
        inverted_shape_dims.remove(axis)
        inverted_shape_dims.append(axis)
        a_along_axis_1d = a.transpose(inverted_shape_dims).flatten().reshape((-1, a.shape[axis]))
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
            weights_along_axis_1d = weights.transpose(inverted_shape_dims).flatten().reshape((-1, weights.shape[axis]))
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
        del out_shape[axis]
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
    return ret, bins_out


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


def nanmean(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> Tuple:
    return jnp.unravel_index(indices, shape)


def quantile(
    a: JaxArray,
    q: Union[float, JaxArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
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
    rowvar: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)


def nanmedian(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )
