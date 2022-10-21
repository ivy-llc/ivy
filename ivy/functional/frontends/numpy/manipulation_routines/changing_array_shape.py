# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape)


def broadcast_to(array, shape, subok=False):
    return ivy.broadcast_to(array, shape)


def ravel(a, order="C"):
    return ivy.reshape(a, (-1,))


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)
