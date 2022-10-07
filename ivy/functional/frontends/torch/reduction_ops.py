import ivy
from collections import namedtuple


def dist(input, other, p=2):
    return ivy.vector_norm(ivy.subtract(input, other), ord=p)


def argmax(input, dim=None, keepdim=False):
    return ivy.argmax(input, axis=dim, keepdims=keepdim)


def argmin(input, dim=None, keepdim=False):
    return ivy.argmin(input, axis=dim, keepdims=keepdim)


def amax(input, dim=None, keepdim=False, *, out=None):
    return ivy.max(input, axis=dim, keepdims=keepdim, out=out)


def amin(input, dim=None, keepdim=False, *, out=None):
    return ivy.min(input, axis=dim, keepdims=keepdim, out=out)


def all(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.all(input, axis=dim, keepdims=keepdim, out=out)
    if ivy.is_uint_dtype(input_dtype):
        ret = ivy.astype(ret, input_dtype, out=out)
    return ret


def any(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.any(input, axis=dim, keepdims=keepdim, out=out)
    if ivy.is_uint_dtype(input_dtype):
        ret = ivy.astype(ret, input_dtype, out=out)
    return ret


def mean(input, dim, keepdim=False, *, out=None):
    return ivy.mean(input, axis=dim, keepdims=keepdim, out=out)


def std(input, dim, unbiased, keepdim=False, *, out=None):
    return ivy.std(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


def prod(input, dim=None, keepdim=False, *, dtype=None, out=None):
    return ivy.prod(input, axis=dim, dtype=dtype, keepdims=keepdim, out=out)


def var(input, dim, unbiased, keepdim=False, *, out=None):
    return ivy.var(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


def min(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        return ivy.min(input, axis=dim, keepdims=keepdim, out=out)
    elif out is not None:
        ivy.min(input, axis=dim, keepdims=keepdim, out=out[0])
        ivy.argmin(input, axis=dim, keepdims=keepdim, out=out[1])
        return out
    else:
        min_tuple = namedtuple("min", ["values", "indices"])
        return min_tuple(
            ivy.min(input, axis=dim, keepdims=keepdim),
            ivy.argmin(input, axis=dim, keepdims=keepdim),
        )
