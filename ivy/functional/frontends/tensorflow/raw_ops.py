# global
import ivy


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis)


concat.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def fill(dims, value, name="full"):
    return ivy.full(dims, value)


fill.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tanh(x, name=None):
    return ivy.tanh(x)


tanh.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
