# global
import ivy


def add(input, other, *, alpha=1, out=None):
    return ivy.add(input, other * alpha, out=out)


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


def atan(input, *, out=None):
    return ivy.atan(input, out=out)


def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


def cos(input, *, out=None):
    return ivy.cos(input, out=out)


def sin(input, *, out=None):
    return ivy.sin(input, out=out)


def acos(input, *, out=None):
    return ivy.acos(input, out=out)


def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


def acosh(input, *, out=None):
    return ivy.acosh(input, out=out)


def arccosh(input, *, out=None):
    return ivy.acosh(input, out=out)


def arccos(input, *, out=None):
    return ivy.acos(input, out=out)


def abs(input, *, out=None):
    return ivy.abs(input, out=out)


def cosh(input, *, out=None):
    return ivy.cosh(input, out=out)


def subtract(input, other, *, alpha=1, out=None):
    return ivy.subtract(input, other * alpha, out=out)


def exp(input, *, out=None):
    return ivy.exp(input, out=out)


def asin(input, *, out=None):
    return ivy.asin(input, out=out)


def arcsin(input, *, out=None):
    return ivy.asin(input, out=out)


def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


def arctanh(input, *, out=None):
    return ivy.atanh(input, out=out)


def log2(input, *, out=None):
    return ivy.log2(input, out=out)


def square(input, *, out=None):
    return ivy.square(input, out=out)


def atan2(input, other, *, out=None):
    return ivy.atan2(input, other, out=out)


def negative(input, *, out=None):
    return ivy.negative(input, out=out)


def bitwise_and(input, other, *, out=None):
    return ivy.bitwise_and(input, other, out=out)


def log10(input, *, out=None):
    return ivy.log10(input, out=out)


def trunc(input, *, out=None):
    return ivy.trunc(input, out=out)


def sqrt(input, *, out=None):
    return ivy.sqrt(input, out=out)


def sign(input, *, out=None):
    return ivy.sign(input, out=out)


def absolute(input, *, out=None):
    return ivy.abs(input, out=out)
