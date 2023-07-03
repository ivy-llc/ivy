import ivy
import numpy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
    handle_numpy_casting,
)
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def percentile(
    x: Union[numpy.ndarray, ivy.Array],
    q: float or sequence of floats,
    axis=None,
    interpolation="linear",
    kind="increasing",
    order="k",
    dtype=None,
    keepdims=False,
):

    x = numpy.asarray(x)

    return ivy.percentile(
        x, q, axis=axis, interpolation=interpolation, kind=kind, order=order, dtype=dtype, keepdims=keepdims
    )
