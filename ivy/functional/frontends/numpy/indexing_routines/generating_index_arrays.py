import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_numpy_arrays,
)


@outputs_to_numpy_arrays
def diag_indices(n, ndim=2):
    idx = ivy.arange(n, dtype=int)
    return (idx,) * ndim


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, k, m)


@to_ivy_arrays_and_back
def tril_indices_from(arr, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return ivy.tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])
