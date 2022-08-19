# global
import ivy


def average(x, axis=None, keepdims=False, name=None):
    return ivy.mean(x)


average.unsupported_dtypes = {"torch": ("float16", "bfload16")}

