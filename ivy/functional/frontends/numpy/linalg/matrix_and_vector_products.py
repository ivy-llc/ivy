# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def outer(a, b, out=None):
    return ivy.outer(a, b, out=out)


@to_ivy_arrays_and_back
def inner(a, b, /):
    return ivy.inner(a, b)


@to_ivy_arrays_and_back
def matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    return ivy.matmul(x1, x2, out=out)


@to_ivy_arrays_and_back
def matrix_power(a, n):
    return ivy.matrix_power(a, n)
