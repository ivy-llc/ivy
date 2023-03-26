from typing import Optional, Union, Tuple, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
=======
    keepdims: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
=======
    keepdims: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
    return jnp.unravel_index(indices.astype(jnp.int32), shape)


def quantile(
    a: JaxArray,
    q: Union[float, JaxArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
<<<<<<< HEAD
    interpolation: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
=======
    interpolation: str = "linear",
    keepdims: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
    rowvar: Optional[bool] = True,
=======
    rowvar: bool = True,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)


def nanmedian(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
=======
    keepdims: bool = False,
    overwrite_input: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
    minlength: Optional[int] = 0,
=======
    minlength: int = 0,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if weights is not None:
        ret = jnp.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = jnp.bincount(x, minlength=minlength).astype(x.dtype)
    return ret
