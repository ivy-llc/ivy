# global
import ivy


def acos(x, name="acos"):
    return ivy.acos(x)


def concat(values, axis, name="concat"):
    return ivy.concat(values, axis)


def fill(dims, value, name="full"):
    return ivy.full(dims, value)
