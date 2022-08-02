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


def hard_sigmoid(x):
    if x < -2.5:
        return 0
    elif x > 2.5:
        return 1
    else:
        return ivy.sum(ivy.multiply(0.2, x), 0.5)


hard_sigmoid.unsupported_dtypes = {"torch": ("float16")}
