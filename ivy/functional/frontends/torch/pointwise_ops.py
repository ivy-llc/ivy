# global
import ivy


def add(input, other, *, alpha=1, out=None):
    return ivy.add(input, other * alpha, out=out)


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


tan.unsupported_dtypes = ("float16",)


def atan(input, *, out=None):
    return ivy.atan(input, out=out)


atan.unsupported_dtypes = ("float16",)


def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


tanh.unsupported_dtypes = ("float16",)


def cos(input, *, out=None):
    return ivy.cos(input, out=out)


cos.unsupported_dtypes = ("float16",)


def sin(input, *, out=None):
    return ivy.sin(input, out=out)


sin.unsupported_dtypes = ("float16",)


def acos(input, *, out=None):
    return ivy.acos(input, out=out)


acos.unsupported_dtypes = ("float16",)


def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


sinh.unsupported_dtypes = ("float16",)


def acosh(input, *, out=None):
    return ivy.acosh(input, out=out)


acosh.unsupported_dtypes = ("float16",)


def arccosh(input, *, out=None):
    return ivy.acosh(input, out=out)


arccosh.unsupported_dtypes = ("float16",)


def arccos(input, *, out=None):
    return ivy.acos(input, out=out)


arccos.unsupported_dtypes = ("float16",)


def abs(input, *, out=None):
    return ivy.abs(input, out=out)


def cosh(input, *, out=None):
    return ivy.cosh(input, out=out)


cosh.unsupported_dtypes = ("float16",)


def subtract(input, other, *, alpha=1, out=None):
    return ivy.subtract(input, other * alpha, out=out)


def exp(input, *, out=None):
    return ivy.exp(input, out=out)


exp.unsupported_dtypes = ("float16",)


def asin(input, *, out=None):
    return ivy.asin(input, out=out)


asin.unsupported_dtypes = ("float16",)


def arcsin(input, *, out=None):
    return ivy.asin(input, out=out)


arcsin.unsupported_dtypes = ("float16",)


def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


asinh.unsupported_dtypes = ("float16",)


def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


atanh.unsupported_dtypes = ("float16",)


def arctanh(input, *, out=None):
    return ivy.atanh(input, out=out)


arctanh.unsupported_dtypes = ("float16",)


def log2(input, *, out=None):
    return ivy.log2(input, out=out)


log2.unsupported_dtypes = ("float16",)


def square(input, *, out=None):
    return ivy.square(input, out=out)


def atan2(input, other, *, out=None):
    return ivy.atan2(input, other, out=out)


atan2.unsupported_dtypes = ("float16",)

