# global
import ivy

from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
)


@to_ivy_arrays_and_back
def outer(a, b, out=None):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.outer(a, b, out=out)


@to_ivy_arrays_and_back
def inner(a, b, /):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.inner(a, b)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    # TODO: add support for the rest of the arguments.
    return ivy.matmul(x1, x2, out=out)


@to_ivy_arrays_and_back
def matrix_power(a, n):
    return ivy.matrix_power(a, n)


@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    return ivy.tensordot(a, b, axes=axes)


@to_ivy_arrays_and_back
def tensorsolve(a, b, axes=2):
    return ivy.tensorsolve(a, b, axes=axes)
