#for review

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
    out: Optional[JaxArray] = None
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.max(a=jnp.asarray(x), axis=axis, keepdims=keepdims)


def mean(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
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
    out: Optional[JaxArray] = None
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.min(a=jnp.asarray(x), axis=axis, keepdims=keepdims)


def prod(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if dtype is None and jnp.issubdtype(x.dtype, jnp.integer):
        if jnp.issubdtype(x.dtype, jnp.signedinteger) and x.dtype in [
            jnp.int8,
            jnp.int16,
            jnp.int32,
        ]:
            dtype = jnp.int32
        elif jnp.issubdtype(x.dtype, jnp.unsignedinteger) and x.dtype in [
            jnp.uint8,
            jnp.uint16,
            jnp.uint32,
        ]:
            dtype = jnp.uint32
        elif x.dtype == jnp.int64:
            dtype = jnp.int64
        else:
            dtype = jnp.uint64
    dtype = ivy.as_native_dtype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[JaxArray] = None
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def sum(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: jnp.dtype = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if dtype is None and jnp.issubdtype(x.dtype, jnp.integer):
        if jnp.issubdtype(x.dtype, jnp.signedinteger) and x.dtype in [
            jnp.int8,
            jnp.int16,
            jnp.int32,
        ]:
            dtype = jnp.int32
        elif jnp.issubdtype(x.dtype, jnp.unsignedinteger) and x.dtype in [
            jnp.uint8,
            jnp.uint16,
            jnp.uint32,
        ]:
            dtype = jnp.uint32
        elif x.dtype == jnp.int64:
            dtype = jnp.int64
        else:
            dtype = jnp.uint64
    dtype = ivy.as_native_dtype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None
) -> JaxArray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims)


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: JaxArray,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.einsum(equation, *operands)
