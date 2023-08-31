import ivy


def as_float_array(X, *, copy=True, force_all_finite=True):
    if ("bool" in str(X.dtype) or "int" in str(X.dtype) or "uint" in str(X.dtype)) and ivy.itemsize(X) <= 4:
        return_dtype = ivy.float32
    else:
        return_dtype = ivy.float64
    return ivy.asarray(X, dtype=return_dtype)
