# global
import ivy
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
def var(x, /, *, axis=None, ddof=0.0, keepdims=False, out=None, dtype=None, where=True):
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = (
        dtype
        if dtype is not None
        else ivy.float64 if ivy.is_int_dtype(x.dtype) else x.dtype
    )
    ret = ivy.var(x, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    ret = (
        ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        if ivy.is_array(where)
        else ret
    )
    return ret.astype(dtype, copy=False)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def mean(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    dtype=None,
    where=True,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))

    ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmean(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    dtype=None,
    where=True,
):
    is_nan = ivy.isnan(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if not ivy.any(is_nan):
        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    else:
        a = [i for i in a if ivy.isnan(i) is False]

        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def std(
    x,
    /,
    *,
    axis=None,
    ddof=0.0,
    keepdims=False,
    out=None,
    dtype=None,
    where=True,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        if ivy.is_int_dtype(x.dtype):
            dtype = ivy.float64
        else:
            dtype = x.dtype
    ret = ivy.std(x, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret.astype(dtype, copy=False)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def average(a, /, *, axis=None, weights=None, returned=False, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    global avg
    avg = 0

    if keepdims is None:
        keepdims_kw = {}
    else:
        keepdims_kw = {"keepdims": keepdims}

    dtype = a.dtype
    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        weights_sum = avg.dtype.type(a.count(axis))
    else:
        if a.shape != weights.shape:
            if axis is None:
                return 0
            weights = ivy.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)
        weights_sum = weights.sum(axis=axis, **keepdims_kw)
        mul = ivy.multiply(a, weights)
        avg = ivy.sum(mul, axis=axis, **keepdims_kw) / weights_sum

    if returned:
        if weights_sum.shape != avg.shape:
            weights_sum = ivy.broadcast_to(weights_sum, avg.shape).copy()
        return avg.astype(dtype), weights_sum
    else:
        return avg.astype(dtype)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanstd(
    a, /, *, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True
):
    a = ivy.nan_to_num(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))

    ret = ivy.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@to_ivy_arrays_and_back
def cov(x, y=None, bias=False, dtype=None, fweights=None, aweights=None, ddof=None):
    # check if inputs are valid
    input_check = ivy.valid_dtype(dtype) and x.ndim in [0, 1]

    if input_check:
        x = ivy.array(x)
        x = x.stack([], axis=0)
        # if two input arrays are given
        if ivy.exists(y) and y.ndim > 0:
            x = x.stack(ivy.array(y), axis=0)

        # compute the weights array
        w = None
        # if weights are 1D and positive
        if ivy.exists(fweights):
            if fweights.ndim < 2 and not fweights.min(keepdims=True)[0] > 0:
                w = ivy.array(fweights)
        if ivy.exists(aweights):
            if aweights.ndim < 2 and not aweights.min(keepdims=True)[0] > 0:
                w = w.multiply(aweights) if ivy.exists(w) else ivy.array(aweights)

            # if w exists, use weighted average
            xw = x.multiply(w)
            w_sum = ivy.sum(w)
            average = ivy.stable_divide(ivy.sum(xw, axis=1), w_sum)
        else:
            # else compute arithmetic average
            average = ivy.mean(x, axis=1)

        # compute the normalization
        if ddof is None:
            ddof = 1 if bias == 0 else 0

        if w is None:
            norm = x.shape[0] - ddof
        elif ddof == 0:
            norm = w_sum
        elif aweights is None:
            norm = w_sum - ddof
        else:
            norm = w_sum - ivy.stable_divide(ddof * ivy.sum(w * aweights), w_sum)

        # compute residuals from average
        x -= average[:]
        # compute transpose matrix
        x_t = ivy.matrix_transpose(x * w) if ivy.exists(w) else ivy.matrix_transpose(x)
        # compute covariance matrix
        c = ivy.stable_divide(ivy.matmul(x, x_t), norm).astype(dtype)

        return c


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    is_nan = ivy.isnan(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if not ivy.any(is_nan):
        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        else:
            dtype = "float" if ivy.is_int_dtype(a) else a.dtype

        ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            where = ivy.array(where, dtype=ivy.bool)
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    else:
        a = [i for i in a if ivy.isnan(i) is False]

        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        else:
            dtype = "float" if ivy.is_int_dtype(a) else a.dtype

        ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            where = ivy.array(where, dtype=ivy.bool)
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    all_nan = ivy.isnan(ret)
    if ivy.all(all_nan):
        ret = ivy.astype(ret, ivy.array([float("inf")]))
    return ret


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


def cpercentile(N, percent, key=lambda x: x):
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
    # print(q)
    if not _quantile_is_valid(q):
        raise ValueError("percentile s must be in the range [0, 100]")
    if axis is None:
        resultarray = []
        nanlessarray = []
        for x in a:
            for i in x:
                if not ivy.isnan(i):
                    nanlessarray.append(i)

        for i in q:
            resultarray.append(cpercentile(nanlessarray, i))
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
                arrayofpercentiles.append(cpercentile(ii, i))
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
                    arrayofpercentiles.append(cpercentile(ii, i))
                resultarray.append(arrayofpercentiles)
        return resultarray


# nanmedian
@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmedian(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    overwrite_input=False,
):
    ret = ivy.nanmedian(
        a, axis=axis, keepdims=keepdims, out=out, overwrite_input=overwrite_input
    )
    return ret
