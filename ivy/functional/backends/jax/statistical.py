# global
import jax.numpy as jnp
from typing import Tuple, Union, Optional, Sequence

# local
import ivy
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# -------------------#


def max(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.max(a=jnp.asarray(x), axis=axis, keepdims=keepdims)


def mean(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def min(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.min(a=jnp.asarray(x), axis=axis, keepdims=keepdims)


def _infer_dtype(dtype: jnp.dtype, x_dtype: jnp.dtype):
    default_dtype = ivy.infer_default_dtype(x_dtype)
    if ivy.dtype_bits(x_dtype) < ivy.dtype_bits(default_dtype):
        dtype = default_dtype
    else:
        dtype = x_dtype
    return dtype


def prod(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(dtype, x.dtype)
    if dtype != x.dtype:
        x = x.astype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def sum(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(dtype, x.dtype)
    if dtype != x.dtype:
        x = x.astype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if isinstance(correction, int):
        return jnp.asarray(
            jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out)
        )
    size = 1
    for a in axis:
        size *= x.shape[a]
    return (size / (size - correction)) * jnp.asarray(
        jnp.var(x, axis=axis, keepdims=keepdims, out=out)
    )


# Extra #
# ------#


def einsum(
    equation: str, *operands: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.einsum(equation, *operands)
