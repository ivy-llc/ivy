# global
import ivy


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


def softmax(input, dim=None):
    return ivy.softmax(input, axis=dim)


softmax.unsupported_dtypes = ("float16",)
