# global
import ivy

from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    from_zero_dim_arrays_to_scalar
)


@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def cross(
    b,
    a,
    /,
    axisa = -1,
    axisb = -1,
    axisc = -1,
    axis  = None
):
    return ivy.cross(a, b, axisa, axisb, axisc, axis)
