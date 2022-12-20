# global
import ivy
from ivy.func_wrapper import from_zero_dim_arrays_to_float
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_float
def percentile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method='linear',
        keepdims=False,
        *,
        interpolation=None
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    ret = ivy.median(a, axis=axis, keepdims=keepdims, out=out)

    return ret
