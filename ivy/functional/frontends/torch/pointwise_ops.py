# global
import ivy


def add(input, other, *, alpha=1, out=None):
    return ivy.add(input, other * alpha, out=out)


add.unsupported_dtypes = ("float16",)


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


tan.unsupported_dtypes = ("float16",)


def abs(input, *, out=None):
    return ivy.abs(input, out=out)


def subtract(input, other, *, alpha=1, out=None):
    return ivy.subtract(input, other * alpha, out=out)
