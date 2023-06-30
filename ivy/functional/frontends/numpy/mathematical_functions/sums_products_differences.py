# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
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
    if ivy.is_array(where):
        x = ivy.where(where, x, ivy.default(out, ivy.zeros_like(x)), out=out)
    if initial is not None:
        s = ivy.to_list(ivy.shape(x, as_array=True))
        s[axis] = 1
        header = ivy.full(ivy.Shape(tuple(s)), initial)
        if ivy.is_array(where):
            x = ivy.where(where, x, ivy.default(out, ivy.zeros_like(x)), out=out)
        x = ivy.concat([header, x], axis=axis)
    else:
        x = ivy.where(ivy.isnan(x), ivy.zeros_like(x), x)
    return ivy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def prod(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    if ivy.is_array(where):
        x = ivy.where(where, x, ivy.default(out, ivy.ones_like(x)), out=out)
    if initial is not None:
        if axis is not None:
            s = ivy.to_list(ivy.shape(x, as_array=True))
            s[axis] = 1
            header = ivy.full(ivy.Shape(tuple(s)), initial)
            x = ivy.concat([header, x], axis=axis)
        else:
            x[0] *= initial
    return ivy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nansum(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):
    fill_values = ivy.zeros_like(a)
    a = ivy.where(ivy.isnan(a), fill_values, a)
    if ivy.is_array(where):
        a = ivy.where(where, a, ivy.default(out, fill_values), out=out)
    if initial is not None:
        a[axis] = 1
        s = ivy.shape(a, as_array=False)
        header = ivy.full(s, initial)
        a = ivy.concat([header, a], axis=axis)
    return ivy.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanprod(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):
    fill_values = ivy.ones_like(a)
    a = ivy.where(ivy.isnan(a), fill_values, a)
    if ivy.is_array(where):
        a = ivy.where(where, a, ivy.default(out, fill_values), out=out)
    if initial is not None:
        a[axis] = 1
        s = ivy.shape(a, as_array=False)
        header = ivy.full(s, initial)
        a = ivy.concat([header, a], axis=axis)
    return ivy.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
def cumsum(a, /, axis=None, dtype=None, out=None):
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
def cumprod(a, /, axis=None, dtype=None, out=None):
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
def nancumprod(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.ones_like(a), a)
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
def nancumsum(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.zeros_like(a), a)
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


@to_ivy_arrays_and_back
def diff(x, /, *, n=1, axis=-1, prepend=None, append=None):
    return ivy.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@to_ivy_arrays_and_back
def ediff1d(ary, to_end=None, to_begin=None):
    diffs = ivy.diff(ary)
    if to_begin is not None:
        if not isinstance(to_begin, (list, tuple)):
            to_begin = [to_begin]
        to_begin = ivy.array(to_begin)
        diffs = ivy.concat((to_begin, diffs))
    if to_end is not None:
        if not isinstance(to_end, (list, tuple)):
            to_end = [to_end]
        to_end = ivy.array(to_end)
        diffs = ivy.concat((diffs, to_end))
    return diffs


@to_ivy_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1):
    return ivy.trapz(y, x=x, dx=dx, axis=axis)
