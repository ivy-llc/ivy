# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x, name=None):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis)


def fill(dims, value, name="full"):
    return ivy.full(dims, value)


fill.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def subtract(x, y, name=None):
    return ivy.subtract(x, y)


subtract.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def difference(x, y, aminusb=True, validate_indices=True): 
    return ivy.difference(x,y)


subtract.unsupported_dtypes = {"torch": ("float16", "bfloat16")}