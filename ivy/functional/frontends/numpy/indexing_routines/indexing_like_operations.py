import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_numpy_arrays,
)


@to_ivy_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@outputs_to_numpy_arrays
def diag_indices(n, ndim=2):
    idx = ivy.arange(n, dtype=int)
    return (idx,) * ndim


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)
