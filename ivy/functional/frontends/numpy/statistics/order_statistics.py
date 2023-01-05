# global
import numpy as np
from numpy import function_base as fb

import ivy.functional.frontends.numpy
from ivy.func_wrapper import infer_dtype
from ivy.functional.frontends.numpy import from_zero_dim_arrays_to_scalar
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@infer_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               *,
               interpolation=None):
    axis = tuple(axis) if isinstance(axis, list) else axis

    methods = ["inverted_cdf", "averaged_inverted_cdf",
               "closest_observation", "interpolated_inverted_cdf",
               "hazen", "weibull", "median_unbiased",
               "normal_unbiased"]
    interpolations = ["linear", "lower", "higher",
                      "midpoint", "nearest"]

    q = np.true_divide(q, 100)
    # undo any decay that the ufunc performed (see gh-13105)
    q = np.asanyarray(q)
    if not fb._quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if interpolation is None:
        if method in methods:
            arr = fb._quantile_unchecked(
                a, q, axis, out, overwrite_input, method, keepdims)
            ret = ivy.infer_dtype(arr)
            return ret
        elif method in interpolations:
            arr = ivy.quantile(a, q, axis=axis, keepdims=keepdims,
                               out=out, interpolation=interpolation)
            ret = ivy.infer_dtype(arr)
            return ret
    elif interpolation is not None:
        if interpolation in interpolations:
            arr = ivy.quantile(a, q, axis=axis, keepdims=keepdims,
                               out=out, interpolation=interpolation)
            ret = ivy.infer_dtype(arr)
            return ret
        elif method in methods:
            arr = fb._quantile_unchecked(
                a, q, axis, out, overwrite_input, method, keepdims)
            ret = ivy.infer_dtype(arr)
            return ret
        elif method in interpolations:
            arr = ivy.quantile(a, q, axis=axis, keepdims=keepdims,
                               out=out, interpolation=interpolation)
            ret = ivy.infer_dtype(arr)
            return ret
