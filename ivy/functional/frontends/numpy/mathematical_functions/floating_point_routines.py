# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_out,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_casting,
)

# This is the nextafter function.
# It returns the next floating-point value after x1 towards x2, element-wise.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _nextafter(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return ivy.nextafter(x1, x2, out=out) # Return result



# This is the spacing function.
# It returns the distance between x and the nearest adjacent number.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _spacing(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return ivy.spacing(x, out=out) # Return result