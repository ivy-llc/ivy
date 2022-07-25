# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x, name=None):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def cos(x, name=None):
    return ivy.cos(x)


cos.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis)


def bitwise_or(x, y, name=None):
    return ivy.bitwise_or(x, y)
