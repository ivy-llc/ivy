# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_float
def sum(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    out=None,
    initial=None,
    where=True,
):
    if not where:
        if dtype:
            return ivy.astype(ivy.array(0), ivy.as_ivy_dtype(dtype))
        return ivy.array(0)
    if initial:
        s = ivy.shape(x, as_array=True)
        s[axis] = 1
        header = ivy.full(ivy.Shape(tuple(s)), initial)
        if where:
            x = ivy.where(where, x, ivy.default(out, ivy.zeros_like(x)))
        x = ivy.concat([x, header], axis=axis)
    return ivy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_float
def prod(
    x, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True
):
    if not where:
        if dtype:
            return ivy.astype(ivy.array(0), ivy.as_ivy_dtype(dtype))
        return ivy.array(0)
    if initial:
        s = ivy.shape(x, as_array=True)
        s[axis] = 1
        header = ivy.full(ivy.Shape(tuple(s)), initial)
        if where:
            x = ivy.where(where, x, ivy.default(out, ivy.ones_like(x)))
        x = ivy.concat([x, header], axis=axis)
    return ivy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def nansum(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):  # ToDo handle initial
    fill_values = ivy.zeros_like(a)
    a = ivy.where(ivy.isnan(a), fill_values, a)
    if where is not None:
        a = ivy.where(where, a, fill_values)
    return ivy.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def nanprod(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):  # ToDo handle initial
    fill_values = ivy.ones_like(a)
    a = ivy.where(ivy.isnan(a), fill_values, a)
    if where is not None:
        a = ivy.where(where, a, fill_values)
    return ivy.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def cumsum(a, /, axis=None, dtype=None, out=None):
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def cumprod(a, /, axis=None, dtype=None, out=None):
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def nancumprod(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.ones_like(a), a)
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def nancumsum(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.zeros_like(a), a)
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)
