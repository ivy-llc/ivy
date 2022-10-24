#global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting_special,
)


@to_ivy_arrays_and_back
def split(a, splits, axis=0):
    return ivy.split(a, splits=None, axis=0)
