import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@to_ivy_arrays_and_back
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def tri(N, M=None, k=0, dtype="float64", *, like=None):
    if M is None:
        M = N
    ones = ivy.ones((N, M), dtype=dtype)
    return ivy.tril(ones, k=k)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)
