# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def argsort(
    x,
    /,
    *,
    axis=-1,
    kind=None,
    order=None,
):
    return ivy.argsort(x, axis=axis)


def sort(a, axis=-1, kind=None, order=None):
    return ivy.sort(a, axis=axis)


def msort(a):
    return ivy.sort(a, axis=0)
