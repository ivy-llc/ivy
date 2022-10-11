# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def roll(a, shift, axis=None):
    return ivy.roll(a, shift, axis=axis)


@to_ivy_arrays_and_back
def flip(m, axis=None):
    return ivy.flip(m, axis=axis, out=None)
