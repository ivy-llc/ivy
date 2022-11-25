import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like,
)
from ivy.exceptions import handle_exceptions


@to_ivy_arrays_and_back
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def diagonal(x, offset, axis1, axis2):
    # ivy.assertions.check_equal(
    #   axis1 == axis2, message="Both axis values should not be the same"
    # )
    # ivy.assertions.check_equal(x.ndim, 2, message="x must be 2-dimensional")

    return ivy.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
