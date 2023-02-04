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
        bins = jnp.linspace(start=range[0], stop=range[1], num=bins + 1, dtype=type(a)).astype(a.dtype)
        range = None
    bins_out = bins.copy()
    if extend_lower_interval:
        bins = bins.at[0].set(-jnp.inf)
    if extend_upper_interval:
        bins = bins.at[-1].set(jnp.inf)
    if axis is not None:
        histogram_values = jnp.apply_along_axis(
            lambda x:
            jnp.histogram(
                a=x,
                bins=bins,
                range=range,
                weights=weights,
            )[0],
            axis,
            a
        )
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = bins_out.astype(dtype)
        return histogram_values, bins_out
    else:
        ret = jnp.histogram(
            a=a,
            bins=bins,
            range=range,
            weights=weights,
            density=density
        )
        histogram_values = ret[0]
        if dtype:
            histogram_values = histogram_values.astype(dtype)
            bins_out = bins_out.astype(dtype)
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
