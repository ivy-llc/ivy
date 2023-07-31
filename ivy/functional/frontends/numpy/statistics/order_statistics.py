# global
import ivy
import numpy
import ivy.numpy as np 
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
    
)

def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (ivy.all(0 <= q) and ivy.all(q <= 1)):
            return False
    return True


def _cpercentile(N, percent, key=lambda x: x):
    """
    Find the percentile   of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile  of the values
    """
    N.sort()
    k = (len(N) - 1) * percent
    f = ivy.math.floor(k)
    c = ivy.math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1
import ivy


@ivy.to_ivy_arrays_and_back
@ivy.handle_numpy_out
def percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False):
    a = ivy.array(a)
    q = ivy.divide(q, 100.0)
    q = ivy.array(q)

    if not _quantile_is_valid(q):
        ivy.logging.warning("percentile s must be in the range [0, 100]")
        return []

    if axis is None:
        resultarray = []
        nanlessarray = ivy.where(~ivy.isnan(a), a, ivy.array(float('nan')))
        for i in q:
            resultarray.append(_cpercentile(nanlessarray, i, interpolation))
        return resultarray
    elif axis == 1:
        resultarray = []
        nanlessarrayofarrays = ivy.stack([ivy.where(~ivy.isnan(row), row, ivy.array(float('nan'))) for row in a])
        for i in q:
            arrayofpercentiles = []
            for ii in nanlessarrayofarrays:
                arrayofpercentiles.append(_cpercentile(ii, i, interpolation))
            resultarray.append(arrayofpercentiles)
        return resultarray
    elif axis == 0:
        try:
            a = ivy.swapaxes(a, 0, 1)
        except ivy.exceptions.IvyError:
            ivy.logging.warning("axis is 0 but couldn't swap")

        finally:
            resultarray = []
            nanlessarrayofarrays = ivy.stack([ivy.where(~ivy.isnan(col), col, ivy.array(float('nan'))) for col in a])
            for i in q:
                arrayofpercentiles = []
                for ii in nanlessarrayofarrays:
                    arrayofpercentiles.append(_cpercentile(ii, i, interpolation))
                resultarray.append(arrayofpercentiles)
            return resultarray

def _quantile_is_valid(q):
    return ivy.all(ivy.logical_and(q >= 0, q <= 1))

def _cpercentile(arr, q, interpolation='linear'):
    # Assuming arr is already nan-less
    sorted_arr = ivy.sort(arr, axis=-1)
    n = ivy.shape(arr)[-1]

    # Calculate percentile index
    idx = q * (n - 1)

    # Split the index into integer and fractional parts
    idx_int = ivy.floor(idx)
    idx_frac = idx - idx_int

    if interpolation == 'lower':
        # Percentile calculated as the lower value
        percentiles = sorted_arr[ivy.minimum(idx_int, n - 1)]
    elif interpolation == 'higher':
        # Percentile calculated as the higher value
        percentiles = sorted_arr[ivy.maximum(idx_int + 1, 0)]
    elif interpolation == 'midpoint':
        # Percentile calculated as the average of two nearest values
        percentiles = 0.5 * (sorted_arr[ivy.minimum(idx_int + 1, n - 1)] + sorted_arr[ivy.minimum(idx_int, n - 1)])
    else:
        # Default interpolation is 'linear'
        lower_vals = ivy.index_update(sorted_arr, idx_int.astype(ivy.int32), 0)
        upper_vals = ivy.index_update(sorted_arr, ivy.clip(idx_int + 1, 0, n - 1).astype(ivy.int32), 0)
        percentiles = (1.0 - idx_frac) * lower_vals + idx_frac * upper_vals

    return percentiles


def nanpercentile(
    a,
    /,
    *,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    interpolation=None,
):
    a = ivy.array(a)
    q = ivy.divide(q, 100.0)
    q = ivy.array(q)

    if not _quantile_is_valid(q):
        # raise ValueError("percentile s must be in the range [0, 100]")
        ivy.logging.warning("percentile s must be in the range [0, 100]")
        return []

    if axis is None:
        resultarray = []
        nanlessarray = []
        for x in a:
            for i in x:
                if not ivy.isnan(i):
                    nanlessarray.append(i)

        for i in q:
            resultarray.append(_cpercentile(nanlessarray, i))
        return resultarray
    elif axis == 1:
        resultarray = []
        nanlessarrayofarrays = []
        for i in a:
            nanlessarray = []
            for t in i:
                if not ivy.isnan(t):
                    nanlessarray.append(t)
            nanlessarrayofarrays.append(nanlessarray)
        for i in q:
            arrayofpercentiles = []
            for ii in nanlessarrayofarrays:
                arrayofpercentiles.append(_cpercentile(ii, i))
            resultarray.append(arrayofpercentiles)
        return resultarray
    elif axis == 0:
        resultarray = []

        try:
            a = ivy.swapaxes(a, 0, 1)
        except ivy.utils.exceptions.IvyError:
            ivy.logging.warning("axis is 0 but couldn't swap")

        finally:
            nanlessarrayofarrays = []
            for i in a:
                nanlessarray = []
                for t in i:
                    if not ivy.isnan(t):
                        nanlessarray.append(t)
                nanlessarrayofarrays.append(nanlessarray)
            for i in q:
                arrayofpercentiles = []
                for ii in nanlessarrayofarrays:
                    arrayofpercentiles.append(_cpercentile(ii, i))
                resultarray.append(arrayofpercentiles)
        return resultarray
