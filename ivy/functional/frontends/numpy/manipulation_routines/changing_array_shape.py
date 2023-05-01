# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)
import sys


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape, order=order)


@to_ivy_arrays_and_back
def resize(x, /, newshape):
    x_new = ravel(x)

    total_size = 1
    for diff_size in newshape:
        total_size *= diff_size
        if diff_size < 0:
            raise ValueError('values must not be negative')
    
    if x_new.size == 0 or total_size == 0:
        return ivy.zeros_like(x_new)   
    
    repetition = -(-total_size//x_new.size)
    # zeros = ivy.zeros((repetition * repetition),dtype=int)
    # x_new = ivy.concat((x_new,zeros))[:total_size]
    # or
    x_new = ivy.concat((x_new,) * repetition)[:total_size]
    y = ivy.reshape(x_new,newshape=newshape,order="C")
    return y


@to_ivy_arrays_and_back
def broadcast_to(array, shape, subok=False):
    return ivy.broadcast_to(array, shape)


@to_ivy_arrays_and_back
def ravel(a, order="C"):
    return ivy.reshape(a, shape=(-1,), order=order)


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)
