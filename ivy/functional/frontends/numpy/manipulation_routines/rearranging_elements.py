# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def flip(m, axis=None):
    return ivy.flip(m, axis=axis, out=None)


@to_ivy_arrays_and_back
def fliplr(m):
    return ivy.fliplr(m, out=None)


@to_ivy_arrays_and_back
def flipud(m):
    return ivy.flipud(m, out=None)


@to_ivy_arrays_and_back
def roll(a, shift, axis=None):
    return ivy.roll(a, shift, axis=axis)


@to_ivy_arrays_and_back
def rot90(m, k=1, axes=(0, 1)):
    return ivy.rot90(m, k=k, axes=axes)
