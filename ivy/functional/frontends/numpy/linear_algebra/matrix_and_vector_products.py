# global
import ivy


def outer(a, b, out=None):
    return ivy.outer(a, b, out=out)


#inner

# global
import ivy


def inner(a, b, out=None):
    return ivy.inner(a, b, out=out)
