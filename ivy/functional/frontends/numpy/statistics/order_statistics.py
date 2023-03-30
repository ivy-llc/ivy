# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_out,
)


@to_ivy_arrays_and_back
# @from_zero_dim_arrays_to_scalar
@handle_numpy_out
def percentile(a,
               q,
               /,
               *,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               interpolation=None):

    axis = tuple(axis) if isinstance(axis, list) else axis

    a = ivy.array(a)

    ret = ivy.percentile(a, q, axis=axis, overwrite_input=overwrite_input,
                         method=method, keep_dims=keepdims, 
                         interpolation=interpolation, out=out)

    return ret