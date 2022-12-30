# global
import ivy.functional.frontends.numpy
from ivy.functional.frontends.numpy import from_zero_dim_arrays_to_scalar
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def quantile(a,
             q,
             axis=None,
             out=None,
             overwrite_input=False,
             method="linear",
             keepdims=False,
             *,
             interpolation=None):
    axis = tuple(axis) if isinstance(axis, list) else axis

    ret = ivy.quantile(a, q, axis=axis, keepdims=keepdims, out=out,
                       interpolation=interpolation)

    return ret
