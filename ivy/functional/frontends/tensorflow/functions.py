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

def l1_l2(l1: float = 0.01, l2: float =0.01):
    return ivy.l1_l2(l1,l2)


fill.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
