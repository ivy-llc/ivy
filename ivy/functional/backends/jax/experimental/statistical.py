from typing import Optional, Union, Tuple, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


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
) -> Tuple[jnp.ndarray]:
    ret = jnp.histogram(
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
