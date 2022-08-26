# global
import ivy


def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper, out=out)


cholesky.unsupported_dtypes = ("float16",)


def ger(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


ger.unsupported_dtypes = {"numpy": ("float16", "int8")}


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


outer.unsupported_dtypes = {"numpy": ("float16", "int8")}
