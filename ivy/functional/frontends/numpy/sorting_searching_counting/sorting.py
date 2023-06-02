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


@to_ivy_arrays_and_back
def sort(a, axis=-1, kind=None, order=None):
    return ivy.sort(a, axis=axis)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


@to_ivy_arrays_and_back
def sort_complex(a):
    return ivy.sort(a)


@to_ivy_arrays_and_back
def lexsort(keys, /, *, axis=-1):
    return ivy.lexsort(keys, axis=axis)


@to_ivy_arrays_and_back
def partition(a, kth, axis=-1, kind="introselect", order=None):
    indices = ivy.argsort(a, axis=axis)
    left = ivy.take_along_axis(a, indices[:kth], axis)
    right = ivy.take_along_axis(a, indices[kth:], axis)
    return ivy.concat((left, right))
