import ivy

"""
This is created for average/mean function. Allows users to accuaretly finds the avearge or mean of n set of values.

"""

novalue=object()
def average_or_mean(a, axis=None, weights=None, returned=False, *,
            keepdims=novalue):

    a = ivy.Array(a)

    if keepdims is novalue:
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        average = a.mean(axis, **keepdims_kw)
        scl = average.dtype.type(a.size/average.size)
    else:
        wgt = ivy.Array(weights)

        if issubclass(a.dtype.type, (int, bool)):
            result_dtype = ivy.result_type(a.dtype, wgt.dtype, 'f8')  #doubt
        else:
            result_dtype = ivy.result_type(a.dtype, wgt.dtype)


        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis should be correctly specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weight is expected when shapes of a and weights differ.")
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

        average= ivy.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != average.shape:
            scl = ivy.broadcast_to(scl, average.shape).copy()
        return average, scl
    else:
        return average