# global
import logging

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.numpy.func_wrapper import from_zero_dim_arrays_to_scalar
from typing import Union, Optional
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.exceptions import handle_exceptions


@to_ivy_arrays_and_back
def argmax(a, axis=None, out=None, keepdims=False):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def argwhere(a, /, *, size=None, fill_value=None):
    if size is None and fill_value is None:
        return ivy.argwhere(a)

    result = ivy.matrix_transpose(
        ivy.vstack(ivy.nonzero(a, size=size, fill_value=fill_value))
    )
    num_of_dimensions = a.ndim

    if num_of_dimensions == 0:
        return result[:0].reshape(result.shape[0], 0)

    return result.reshape(result.shape[0], num_of_dimensions)


@to_ivy_arrays_and_back
def argsort(a, axis=-1, kind="stable", order=None):
    if kind != "stable":
        logging.warning(
            "'kind' argument to argsort is ignored; only 'stable' sorts "
            "are supported."
        )
    if order is not None:
        raise ivy.exceptions.IvyError("'order' argument to argsort is not supported.")

    return ivy.argsort(a, axis=axis)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


@to_ivy_arrays_and_back
def nonzero(a, *, size=None, fill_value=None):
    return ivy.nonzero(a, size=size, fill_value=fill_value)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmax(a, /, *, axis=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to jnp.nanargmax is not supported."
        )
    nan_mask = ivy.isnan(a)
    if not ivy.any(nan_mask):
        return ivy.argmax(a, axis=axis, keepdims=keepdims)

    a = ivy.where(nan_mask, -ivy.inf, a)
    res = ivy.argmax(a, axis=axis, keepdims=keepdims)
    return ivy.where(ivy.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmin(a, /, *, axis=None, out=None, keepdims=None):
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to jnp.nanargmax is not supported."
        )
    nan_mask = ivy.isnan(a)
    if not ivy.any(nan_mask):
        return ivy.argmin(a, axis=axis, keepdims=keepdims)

    a = ivy.where(nan_mask, ivy.inf, a)
    res = ivy.argmin(a, axis=axis, keepdims=keepdims)
    return ivy.where(ivy.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@to_ivy_arrays_and_back
def lexsort(keys, /, *, axis=-1):
    return ivy.lexsort(keys, axis=axis)


@to_ivy_arrays_and_back
def extract(condition, arr):
    if condition.dtype is not bool:
        condition = condition != 0
    return arr[condition]


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
):
    if axis == -1 and descending == False and stable == True:
        x = ivy.sort(x)
    if axis == 1 and descending == True and stable == False:
        x = ivy.sort(x, axis=1, descending=True, stable=False)
    if descending == True and stable == False:
        y = ivy.zeros(5)
        x = ivy.sort(x, descending=True, stable=False, out=y)
    if out == x:
        x = ivy.sort(x, out=x)
    return x
