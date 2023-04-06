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


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    if ivy.is_float_dtype(x):
        x = x.astype(ivy.float64)
    elif ivy.is_bool_dtype or ivy.is_int_dtype(x):
        x = x.astype(ivy.int64)
    return ivy.vander(x, N=N, increasing=increasing)


# diagflat
def diagflat(v, offset=0, padding_value=0, align="RIGHT_LEFT", num_rows=-1, num_cols=-1):
    return ivy.diagflat(v, offset=offset, padding_value=padding_value, num_rows=num_rows, num_cols=num_cols)
