import ivy
from collections import namedtuple


def argmax(input, dim=None, keepdim=False):
    return ivy.argmax(input, axis=dim, keepdims=keepdim)


def argmin(input, dim=None, keepdim=False):
    return ivy.argmin(input, axis=dim, keepdims=keepdim)


def amax(input, dim, keepdim=False, *, out=None):
    return ivy.max(input, axis=dim, keepdims=keepdim, out=out)


def amin(input, dim, keepdim=False, *, out=None):
    return ivy.min(input, axis=dim, keepdims=keepdim, out=out)


def aminmax(input, *, dim=None, keepdim=False, out=None):
    out_arrays = out is not None
    if out_arrays:
        if not isinstance(out_arrays, tuple):
            raise TypeError("out must be a tuple")
        if len(out) != 2:
            raise ValueError("out must be a tuple of length 2")

    min_array = out[0] if out_arrays else None
    max_array = out[1] if out_arrays else None
    tuple_key = "aminmax_out" if out_arrays else "aminmax"

    min_out = ivy.min(input, axis=dim, keepdims=keepdim, out=min_array)

    max_out = ivy.max(input, axis=dim, keepdims=keepdim, out=max_array)

    ret = namedtuple(tuple_key, ["min", "max"])(min_out, max_out)

    return ret


aminmax.unsupported_dtypes = ("float16",)


def all(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.all(input, axis=dim, keepdims=keepdim, out=out)
    if input_dtype == ivy.uint8:
        ret = ivy.astype(ret, ivy.uint8, out=out)
    return ret


def any(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.any(input, axis=dim, keepdims=keepdim, out=out)
    if input_dtype == ivy.uint8:
        ret = ivy.astype(ret, ivy.uint8, out=out)
    return ret


def max(input, dim=None, keepdim=False, *, out=None):
    out_arrays = out is not None
    if out_arrays:
        if not isinstance(out_arrays, tuple):
            raise TypeError("out must be a tuple")
        if len(out) != 2:
            raise ValueError("out must be a tuple of length 2")

    values_array = out[0] if out_arrays else None
    indices_array = out[1] if out_arrays else None
    tuple_key = "max_out" if out is not None else "max"

    values = ivy.max(input, axis=dim, keepdims=keepdim, out=values_array)
    indices = ivy.argmax(input, axis=dim, keepdims=keepdim, out=indices_array)

    ret = namedtuple(tuple_key, ["values", "indices"])(values, indices)

    return ret


def min(input, dim=None, keepdim=False, *, out=None):
    out_arrays = out is not None
    if out_arrays:
        if not isinstance(out_arrays, tuple):
            raise TypeError("out must be a tuple")
        if len(out) != 2:
            raise ValueError("out must be a tuple of length 2")

    values_array = out[0] if out_arrays else None
    indices_array = out[1] if out_arrays else None
    tuple_key = "min_out" if out is not None else "min"

    values = ivy.min(input, axis=dim, keepdims=keepdim, out=values_array)
    indices = ivy.argmin(input, axis=dim, keepdims=keepdim, out=indices_array)

    tuple_key = "min_out" if out is not None else "min"

    ret = namedtuple(tuple_key, ["values", "indices"])(values, indices)

    return ret
