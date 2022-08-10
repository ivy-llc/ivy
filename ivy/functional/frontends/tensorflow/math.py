# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x, name=None):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def multiply(x, y, name=None):
    return ivy.multiply(x, y)


multiply.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def subtract(x, y, name=None):
    return ivy.subtract(x, y)


subtract.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def logical_xor(x, y, name='LogicalXor'):
    return ivy.logical_xor(x, y)


logical_xor.supported_dtypes = {"torch": ("bool", "bool")}


def divide(x, y, name=None):
    return ivy.divide(x, y)


divide.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

def logical_or(x, y, name='LogicalOr'):
    return ivy.logical_or(x, y)

logical_or.supported_dtypes = {"torch": ("bool", "bool")}
