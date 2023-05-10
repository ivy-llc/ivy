# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)
import sys
import numpy as np

@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape, order=order)


@to_ivy_arrays_and_back
def resize(x, newshape,/,refcheck=True):
    x_new = np.ravel(x)
    total_size = 1
    for diff_size in newshape:
        total_size *= diff_size
        if diff_size < 0:
            raise ValueError('values must not be negative')
    
    if x_new.size == 0 or total_size == 0:
        return np.zeros_like(x_new) 
    repetition = -(-total_size//len(x_new))
    # zeros = ivy.zeros((repetition * repetition),dtype=int)
    # x_new = ivy.concat((x_new,zeros))[:total_size]
    # or
    conc = (x_new,) * repetition
    x_new = np.concatenate(conc)[:total_size]
    y = np.reshape(x_new,newshape=newshape,order="C")
    return ivy.array(y)


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

