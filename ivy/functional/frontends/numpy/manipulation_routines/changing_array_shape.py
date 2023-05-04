# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape, order=order)


@to_ivy_arrays_and_back
def resize(x, /, newshape, refcheck=True):
    return ivy.resize(x, newshape=newshape, refcheck=refcheck)


@to_ivy_arrays_and_back
def broadcast_to(array, shape, subok=False):
    return ivy.broadcast_to(array, shape)


@to_ivy_arrays_and_back
def ravel(a, order="C"):
    return ivy.reshape(a, shape=(-1,), order=order)


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)


@to_ivy_arrays_and_back
def asarray_chkfinite(a, dtype=None, order=None):
    a = ivy.asarray(a, dtype=dtype)
    if not ivy.all(ivy.isfinite(a)):
        raise ValueError("array must not contain infs or NaNs")
    return a
