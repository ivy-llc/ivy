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
    return ivy.nextafter(x1, x2, out=out)
