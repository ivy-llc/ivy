from typing import Optional, Union, Tuple, Sequence
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
    bins: Optional[Union[int, Sequence[int], str]] = None,
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
        bins[0] = -jnp.inf
    if extend_upper_interval:
        bins[-1] = jnp.inf
    if axis is None:
        axis = 0
    if a.ndim > 0:
        if weights is not None:
            a_is, a_ks = a.shape[:axis], a.shape[axis + 1:]
            weights_is, weights_ks = weights.shape[:axis], weights.shape[axis + 1:]
            out_shape = list(a.shape)
            out_shape[axis] = jnp.array(bins).size - 1
            histogram_values = jnp.zeros(shape=out_shape)
            for a_i, weights_i in zip(jnp.ndindex(a_is), jnp.ndindex(weights_is)):
                for a_k, weights_k in zip(jnp.ndindex(a_ks), jnp.ndindex(weights_ks)):
                    f = jnp.histogram(
                        a[a_i + jnp.s_[:, ] + a_k],
                        bins=bins,
                        range=range,
                        weights=weights[weights_i + jnp.s_[:, ] + weights_k],
                    )[0]
                    f_js = f.shape
                    for f_j in jnp.ndindex(f_js):
                        histogram_values[a_i + f_j + a_k] = f[f_j]
        else:
            histogram_values = jnp.apply_along_axis(
                lambda x: jnp.histogram(
                    a=x,
                    bins=bins,
                    range=range,
                )[0],
                axis,
                a,
            )
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = jnp.array(bins_out).astype(dtype)
        return histogram_values, bins_out
    else:
        ret = jnp.histogram(
            a=a, bins=bins, range=range, weights=weights, density=density
        )
        histogram_values = ret[0]
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = jnp.array(bins_out).astype(dtype)
        return histogram_values, bins_out


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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
    return jnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.unravel_index(indices, shape)


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
    rowvar: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)
