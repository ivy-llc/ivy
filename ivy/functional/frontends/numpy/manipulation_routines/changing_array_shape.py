# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def asanyarray(a, dtype=None, order=None, like=None):
    return ivy.asarray(a)


@to_ivy_arrays_and_back
def asarray_chkfinite(a, dtype=None, order=None):
    a = ivy.asarray(a, dtype=dtype)
    if not ivy.all(ivy.isfinite(a)):
        raise ValueError("array must not contain infs or NaNs")
    return a


@to_ivy_arrays_and_back
def asfarray(a, dtype=ivy.float64):
    return ivy.asarray(a, dtype=ivy.float64)


@to_ivy_arrays_and_back
def broadcast_to(array, shape, subok=False):
    return ivy.broadcast_to(array, shape)


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)


@to_ivy_arrays_and_back
def ravel(a, order="C"):
    return ivy.reshape(a, shape=(-1,), order=order)


@to_ivy_arrays_and_back
def require(a, dtype=None, requirements=None, *, like=None):
    return ivy.asarray(a, dtype=dtype)


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape, order=order)


@to_ivy_arrays_and_back
def resize(x, newshape, /, refcheck=True):
    if isinstance(newshape, int):
        newshape = (newshape,)
    x_new = ivy.reshape(x, shape=(-1,), order="C")
    total_size = 1
    for diff_size in newshape:
        total_size *= diff_size
        if diff_size < 0:
            raise ValueError("values must not be negative")
    if x_new.size == 0 or total_size == 0:
        return ivy.zeros_like(x_new)
    repetition = -(-total_size // len(x_new))
    conc = (x_new,) * repetition
    x_new = ivy.concat(conc)[:total_size]
    y = ivy.reshape(x_new, shape=newshape, order="C")
    return y
