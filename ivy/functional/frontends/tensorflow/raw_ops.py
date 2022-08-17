# global
import ivy


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis)


def fill(dims, value, name="full"):
    return ivy.full(dims, value)


def linear(x, name="linear"):
        return ivy.linear(x)


    linear.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
