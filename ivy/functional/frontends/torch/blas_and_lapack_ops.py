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


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


outer.unsupported_dtypes = {"numpy": ("float16", "int8")}


def dot(input, other, *, out=None):
    return ivy.vecdot(input, other, out=out)

dot.unsupported_dtypes = {"numpy": ("float16", "int8")}