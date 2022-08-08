# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x, name=None):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
