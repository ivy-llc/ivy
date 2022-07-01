# global
import ivy


def add(x, y):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
