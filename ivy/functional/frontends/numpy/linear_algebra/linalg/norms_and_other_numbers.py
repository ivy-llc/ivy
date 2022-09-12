# local
import ivy
from ivy.func_wrapper import from_zero_dim_arrays_to_float


def norm(x, ord=None, axis=None, keepdims=False):
    ret = ivy.vector_norm(x, axis, keepdims, ord)
    if axis is None:
        return ret[0]
    return ret


norm.unsupported_dtypes = ("float16",)


# matrix_rank
def matrix_rank(A, tol=None, hermitian=False):
    ret = ivy.matrix_rank(A, rtol=tol)
    return ivy.array(ret, dtype=ivy.int64)


# det
def det(a):
    return ivy.det(a)


# slogdet
@from_zero_dim_arrays_to_float
def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    return sign, logabsdet
