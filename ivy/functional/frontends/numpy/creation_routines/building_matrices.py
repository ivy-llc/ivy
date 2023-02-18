import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@with_unsupported_dtypes({"1.23.0 and below": ("float32", "float16")}, "numpy")
@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    if N == 0:
        return ivy.array([], dtype=x.dtype)
    else:
        return ivy.vander(x, N=N, increasing=increasing, out=None)
