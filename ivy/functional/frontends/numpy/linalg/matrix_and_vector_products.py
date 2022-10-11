# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_casting
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

def outer(a, b, out=None):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.outer(a, b, out=out)


def inner(a, b, /):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.inner(a, b)


@handle_numpy_casting
def matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    return ivy.matmul(x1, x2, out=out)


def matrix_power(a, n):
    return ivy.matrix_power(a, n)
