# global
import ivy


def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper, out=out)


cholesky.unsupported_dtypes = ("float16",)


def ger(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


ger.unsupported_dtypes = {"numpy": ("float16", "int8")}


def inverse(input, *, out=None):
    return ivy.inv(input, out=out)


inverse.unsupported_dtypes = ("float16",)


def det(input):
    return ivy.det(input)


det.unsupported_dtypes = ("float16",)


def slogdet(input):
    return ivy.slogdet(input)


slogdet.unsupported_dtypes = ("float16",)


def matmul(input, other, *, out=None):
    return ivy.matmul(input, other, out=out)


def mm():
    pass


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


outer.unsupported_dtypes = {"numpy": ("float16", "int8")}


def pinverse():
    pass


def qr():
    pass


def solve():
    pass


def svd():
    pass
