import ivy
import numpy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
    
)
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar

def percentile(
    a,
    q,
    axis=None,
    interpolation='linear',
    keepdims=False
):
    a = ivy.array(a)
    q = ivy.divide(q, 100.0)
    q = ivy.array(q)

    if not ivy.all(ivy.logical_and(q >= 0, q <= 1)):
        ivy.logging.warning("Percentiles must be in the range [0, 100]")
        return []

    if axis is None:
        nanlessarray = ivy.where(ivy.isnan(a), ivy.inf, a)
        nanlessarray = ivy.reshape(nanlessarray, (-1,))
        resultarray = ivy.percentile(nanlessarray, q)
        return resultarray
    elif axis == 1:
        nanlessarrayofarrays = ivy.where(ivy.isnan(a), ivy.inf, a)
        resultarray = ivy.percentile(nanlessarrayofarrays, q, axis=1, keepdims=keepdims)
        return resultarray
    elif axis == 0:
        nanlessarrayofarrays = ivy.where(ivy.isnan(a), ivy.inf, a)
        nanlessarrayofarrays = ivy.swapaxes(nanlessarrayofarrays, 0, 1)
        resultarray = ivy.percentile(nanlessarrayofarrays, q, axis=0, keepdims=keepdims)
        return resultarray
