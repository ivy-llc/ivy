import ivy


def accumulate(method, array, axis=0, dtype=None, out=None):
    return array.__array_ufunc__(method, "accummulate", array=array,
        axis=axis, dtype=dtype, out=out)
