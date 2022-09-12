# global
import ivy


def outer(a, b, out=None):
    return ivy.outer(a, b, out=out)


def inner(a, b, out=None):
    return ivy.inner(a, b, out=out)
