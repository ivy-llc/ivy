# local
import ivy
import numpy as np
from typing import Tuple
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape, order=order)


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
def resize(a: np.ndarray, new_shape: Tuple[int]):
    if isinstance(new_shape, int):
        new_shape = (new_shape,)

    a = ravel(a)

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")

    if np.prod(a.shape) == 0 or new_size == 0:
        return np.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // np.prod(a.shape))
    if repeats != 1:
        a = np.concatenate((a,) * repeats)[:new_size]
    return np.reshape(a, new_shape)
