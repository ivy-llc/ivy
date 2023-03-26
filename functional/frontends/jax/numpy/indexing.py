# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def diagonal(a, offset=0, axis1=0, axis2=1):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def diag_indices(n, ndim=2):
    idx = ivy.arange(n, dtype=int)
    return (idx,) * ndim


# take_along_axis
@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis, mode="fill"):
    return ivy.take_along_axis(arr, indices, axis, mode=mode)
<<<<<<< HEAD
=======


@to_ivy_arrays_and_back
def tril_indices(n_rows, n_cols=None, k=0):
    return ivy.tril_indices(n_rows, n_cols, k)
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
