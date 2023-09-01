import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


# diagflat
@to_ivy_arrays_and_back
def diagflat(v, k=0):
    ret = ivy.diagflat(v, offset=k)
    while len(ivy.shape(ret)) < 2:
        ret = ret.expand_dims(axis=0)
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
def tri(N, M=None, k=0, dtype="float64", *, like=None):
    if M is None:
        M = N
    ones = ivy.ones((N, M), dtype=dtype)
    return ivy.tril(ones, k=k)


@to_ivy_arrays_and_back
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    if ivy.is_float_dtype(x):
        x = x.astype(ivy.float64)
    elif ivy.is_bool_dtype or ivy.is_int_dtype(x):
        x = x.astype(ivy.int64)
    return ivy.vander(x, N=N, increasing=increasing)
