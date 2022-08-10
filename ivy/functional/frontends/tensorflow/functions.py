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


def multiply(x, y, name=None):
    return ivy.multiply(x, y)


multiply.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def subtract(x, y, name=None):
    return ivy.subtract(x, y)


subtract.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def logical_xor(x, y, name='LogicalXor'):
    return ivy.logical_xor(x, y)


logical_xor.supported_dtypes = {"torch": ("bool", "bool")}


def logical_and(x, y, name='LogicalAnd'):
    return ivy.logical_and(x, y)


logical_and.supported_dtypes = {"torch": ("bool", "bool")}