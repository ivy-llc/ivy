from numbers import Number
from typing import Optional, Tuple, Union

import jax.numpy as jnp

import ivy
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# ------------------ #


def argmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.argmax(x, axis=axis, keepdims=keepdims)
    if output_dtype:
        ret = ret.astype(output_dtype)
    return ret


def argmin(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.argmin(x, axis=axis, keepdims=keepdims)
    # The returned array must have the default array index data type.
    if output_dtype is not None:
        output_dtype = ivy.as_native_dtype(output_dtype)
        if output_dtype not in (jnp.int32, jnp.int64):
            return jnp.array(ret, dtype=jnp.int64)
        else:
            return jnp.array(ret, dtype=output_dtype)
    return jnp.array(ret, dtype=jnp.int64)


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


def argwhere(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.argwhere(x)
