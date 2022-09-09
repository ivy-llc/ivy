# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


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


def cumsum(a, /, axis=None, dtype=None, out=None):
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


def cumprod(a, /, axis=None, dtype=None, out=None):
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


def nancumprod(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.ones_like(a), a)
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


def nancumsum(a, /, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.zeros_like(a), a)
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)
