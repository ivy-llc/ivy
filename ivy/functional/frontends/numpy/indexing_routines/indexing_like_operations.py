import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis):
    return ivy.take_along_axis(arr, indices, axis)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
