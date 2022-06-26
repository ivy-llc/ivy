# global
import ivy


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


tan.unsupported_dtypes = ('float16',)


def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)
