from typing import Optional, Union, Tuple, Sequence

from ivy import with_unsupported_dtypes
from . import backend_version
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


@with_unsupported_dtypes(
    {"1.11.0 and below": ("complex64", "complex128")}, backend_version
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
    return jnp.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
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
