# global
import ivy
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def mean(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))

    ret = ivy.mean(x, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


mean.unsupported_dtypes = {"torch": ("float16",)}

novalue=object()
def average(a, axis=None, weights=None, returned=False, *,
            keepdims=novalue):

    a = ivy.Array(a)

    if keepdims is novalue:
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        scl = avg.dtype.type(a.size/avg.size)
    else:
        wgt = ivy.Array(weights)

        if issubclass(a.dtype.type, (int, bool)):
            result_dtype = ivy.result_type(a.dtype, wgt.dtype, 'f8')  #doubt
        else:
            result_dtype = ivy.result_type(a.dtype, wgt.dtype)


        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = ivy.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = ivy.swapaxes(wgt,-1, axis)

        scl = ivy.sum(wgt,axis=axis, dtype=result_dtype)
        if ivy.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = ivy.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = ivy.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg

