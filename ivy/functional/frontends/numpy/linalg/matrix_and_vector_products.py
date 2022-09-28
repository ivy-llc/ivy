# global
import ivy


def outer(a, b, out=None):
    return ivy.outer(a, b, out=out)


def inner(a, b, /):
    return ivy.inner(a, b)


def matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    return ivy.matmul(x1, x2, out=out)


def matrix_power(a, n):
    return ivy.matrix_power(a, n)
