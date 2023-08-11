# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
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


@to_ivy_arrays_and_back
@handle_numpy_out
def ptp(a, axis=None, out=None, keepdims=False):
    x = ivy.max(a, axis=axis, keepdims=keepdims)
    y = ivy.min(a, axis=axis, keepdims=keepdims)
    ret = ivy.subtract(x, y)
    return ret.astype(a.dtype, copy=False)


@handle_numpy_out
def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    values = ivy.array(a)
    quantile = ivy.divide(q, 100.0)
    quantile_arr = ivy.array(quantile)

    if quantile_arr.ndim == 0:
        quantile_arr = ivy.array([quantile])

    if not _quantile_is_valid(quantile_arr):
        # raise ValueError("Percentiles must be in the range of [0, 100].")
        ivy.logging.warning("Percentiles must be in the range of [0, 100].")
        return []

    output = []

    if axis is None:
        if ivy.any(ivy.isnan(values)):
            output = [ivy.nan for _ in quantile_arr]

        else:
            reshaped_arr = ivy.sort(ivy.reshape(values, -1))

            output = [_cpercentile(reshaped_arr, quantile) for quantile in quantile_arr]

    elif axis == 0:
        for quantile in quantile_arr:
            q_row = []
            for col_idx in range(values.shape[1]):
                if ivy.any(ivy.isnan(values[:, col_idx])):
                    val = ivy.nan
                else:
                    val = _cpercentile(ivy.sort(values[:, col_idx]), quantile)

                q_row.append(val)

            output.append(q_row)

    elif axis == 1:
        for quantile in quantile_arr:
            q_row = []
            for row_idx in range(values.shape[0]):
                if ivy.any(ivy.isnan(values[row_idx, :])):
                    val = ivy.nan
                else:
                    val = _cpercentile(ivy.sort(values[row_idx, :]), quantile)
                q_row.append(val)

            output.append(q_row)

    return output[0] if output.shape[0] == 1 else output


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
