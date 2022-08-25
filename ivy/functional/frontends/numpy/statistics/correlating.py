import ivy


def sum(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    return ivy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


sum.unsupported_dtypes = {"torch": ("float16",)}
