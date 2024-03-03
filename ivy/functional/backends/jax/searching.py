from numbers import Number
from typing import Optional, Tuple, Union

import jax.numpy as jnp

import ivy
from ivy.functional.backends.jax import JaxArray
from . import backend_version
from ivy.func_wrapper import with_unsupported_dtypes

# Array API Standard #
# ------------------ #


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def argmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if select_last_index:
        x = jnp.flip(x, axis=axis)
        ret = jnp.array(jnp.argmax(x, axis=axis, keepdims=keepdims))
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = x.size - ret - 1
    else:
        ret = jnp.argmax(x, axis=axis, keepdims=keepdims)
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.astype(dtype)
    return ret


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def argmin(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[jnp.dtype] = None,
    select_last_index: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if select_last_index:
        x = jnp.flip(x, axis=axis)
        ret = jnp.array(jnp.argmin(x, axis=axis, keepdims=keepdims))
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = x.size - ret - 1
    else:
        ret = jnp.argmin(x, axis=axis, keepdims=keepdims)
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.astype(dtype)
    return ret


def nonzero(
    x: JaxArray,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[JaxArray, Tuple[JaxArray]]:
    res = jnp.nonzero(x, size=size, fill_value=fill_value)

    if as_tuple:
        return tuple(res)

    return jnp.stack(res, axis=1)


def where(
    condition: JaxArray,
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ivy.astype(jnp.where(condition, x1, x2), x1.dtype, copy=False)


# Extra #
# ----- #


def argwhere(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.argwhere(x)
