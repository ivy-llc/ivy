# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@to_ivy_arrays_and_back
def nextafter(x1, x2, /, *, out=None):
    return ivy.nextafter(x1, x2, out=out)
