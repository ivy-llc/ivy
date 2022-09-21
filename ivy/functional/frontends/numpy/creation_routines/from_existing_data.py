import ivy

def asarray(a, dtype=None, order=None, /, like=None):
    if dtype:
        return ivy.asarray(a, dtype=dtype)
    return ivy.asarray(a, dtype=a.dtype())
