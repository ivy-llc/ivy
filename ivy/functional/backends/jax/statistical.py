# global
import jax
import jax.numpy as jnp
from typing import Union, Optional, Sequence


# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version

# Array API Standard #
# -------------------#


def min(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.min(
        a=jnp.asarray(x), axis=axis, keepdims=keepdims, initial=initial, where=where
    )


def max(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.max(a=jnp.asarray(x), axis=axis, keepdims=keepdims)


@with_unsupported_dtypes(
    {"0.4.24 and below": "bfloat16"},
    backend_version,
)
def mean(
    x: JaxArray,
    /,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is not None:
        dtype = ivy.as_native_dtype(dtype)
        x = jnp.astype(x, dtype)
    return jnp.mean(x, axis=axis, keepdims=keepdims, dtype=x.dtype)


def _infer_dtype(dtype: jnp.dtype):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
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
        dtype = x.dtype
    if dtype != x.dtype and not ivy.is_bool_dtype(x):
        x = jnp.astype(x, dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    if ivy.is_bool_dtype(x):
        if jax.config.jax_enable_x64:
            dtype = ivy.as_native_dtype("int64")
        else:
            dtype = ivy.as_native_dtype("int32")
    return jnp.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if isinstance(correction, int):
        ret = jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out)
        return ivy.astype(ret, x.dtype, copy=False)
    if x.size == 0:
        return jnp.asarray(float("nan"))
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size == correction:
        size += 0.0001  # to avoid division by zero in return
    return ivy.astype(
        jnp.multiply(
            jnp.var(x, axis=axis, keepdims=keepdims, out=out),
            size / jnp.abs(size - correction),
        ),
        x.dtype,
        copy=False,
    )


# Extra #
# ------#


@with_unsupported_dtypes({"0.4.24 and below": ("bfloat16", "bool")}, backend_version)
def cumprod(
    x: JaxArray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is jnp.bool_:
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if not (exclusive or reverse):
        return jnp.cumprod(x, axis, dtype=dtype)
    elif exclusive and reverse:
        x = jnp.cumprod(jnp.flip(x, axis=(axis,)), axis=axis, dtype=dtype)
        x = jnp.swapaxes(x, axis, -1)
        x = jnp.concatenate((jnp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = jnp.swapaxes(x, axis, -1)
        return jnp.flip(x, axis=(axis,))

    elif exclusive:
        x = jnp.swapaxes(x, axis, -1)
        x = jnp.concatenate((jnp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = jnp.cumprod(x, -1, dtype=dtype)
        return jnp.swapaxes(x, axis, -1)
    else:
        x = jnp.cumprod(jnp.flip(x, axis=(axis,)), axis=axis, dtype=dtype)
        return jnp.flip(x, axis=axis)


@with_unsupported_dtypes({"0.4.24 and below": "bool"}, backend_version)
def cumsum(
    x: JaxArray,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is jnp.bool_:
            dtype = ivy.default_int_dtype(as_native=True)
        elif ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        else:
            dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = jnp.cumsum(jnp.flip(x, axis=axis), axis=axis, dtype=dtype)
            x = jnp.swapaxes(x, axis, -1)
            x = jnp.concatenate((jnp.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = jnp.swapaxes(x, axis, -1)
            res = jnp.flip(x, axis=axis)
        elif exclusive:
            x = jnp.swapaxes(x, axis, -1)
            x = jnp.concatenate((jnp.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = jnp.cumsum(x, -1, dtype=dtype)
            res = jnp.swapaxes(x, axis, -1)
        elif reverse:
            x = jnp.cumsum(jnp.flip(x, axis=axis), axis=axis, dtype=dtype)
            res = jnp.flip(x, axis=axis)
        return res
    return jnp.cumsum(x, axis, dtype=dtype)


def einsum(
    equation: str, *operands: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.einsum(equation, *operands)
