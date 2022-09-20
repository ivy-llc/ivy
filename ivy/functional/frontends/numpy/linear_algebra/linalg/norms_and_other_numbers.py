# local
import ivy
import ivy.functional.frontends as frontends
from ivy.func_wrapper import with_unsupported_dtypes

versions = frontends.versions["numpy"]


# solve
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, versions)
def norm(x, ord=None, axis=None, keepdims=False):
    ret = ivy.vector_norm(x, axis, keepdims, ord)
    if axis is None:
        return ret[0]
    return ret


# matrix_rank
def matrix_rank(A, tol=None, hermitian=False):
    ret = ivy.matrix_rank(A, rtol=tol)
    return ivy.array(ret, dtype=ivy.int64)


# det
def det(a):
    return ivy.det(a)


# slogdet
def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    return ivy.concat((ivy.reshape(sign, (-1,)), ivy.reshape(logabsdet, (-1,))))
