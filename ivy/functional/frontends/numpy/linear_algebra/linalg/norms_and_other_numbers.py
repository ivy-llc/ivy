# global
import ivy

# matrix_rank


def matrix_rank(A, tol=None, hermitian=False):
    ret = ivy.matrix_rank(x=A, rtol=tol)
    return ivy.array(ret, dtype=ivy.int64)


matrix_rank.unsupported_dtypes = ("float16",)

# det


def det(a):
    return ivy.det(a)


det.unsupported_dtypes = ("float16",)

# slogdet


def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    ret = ivy.concat((ivy.reshape(sign, (-1,)), ivy.reshape(logabsdet, (-1,))))
    return ret


slogdet.unsupported_dtypes = ("float16",)
