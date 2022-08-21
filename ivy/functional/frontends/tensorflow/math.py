# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


def tan(x, name=None):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16",)}


def multiply(x, y, name=None):
    return ivy.multiply(x, y)


def subtract(x, y, name=None):
    return ivy.subtract(x, y)


def logical_xor(x, y, name="LogicalXor"):
    return ivy.logical_xor(x, y)


logical_xor.supported_dtypes = {"torch": ("bool", "bool")}


def divide(x, y, name=None):
    return ivy.divide(x, y)


def negative(x, name=None):
    return ivy.negative(x)


negative.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def log_sigmoid(x, name=None):
    return -ivy.softplus(-x)


log_sigmoid.unsupported_dtypes = {
    "torch": ("float16", "bfloat16"),
    "numpy": ("float16", "bfloat16", "float32", "float64"),
}


def reduce_max(input_tensor, axis=None, keepdims=False, name="reduce_max"):
    return ivy.max(input_tensor, axis=axis, keepdims=keepdims)


reduce_max.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reduce_min(input_tensor, axis=None, keepdims=False, name="reduce_min"):
    return ivy.min(input_tensor, axis=axis, keepdims=keepdims)


reduce_min.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reduce_prod(input_tensor, axis=None, keepdims=False, name="reduce_prod"):
    return ivy.prod(input_tensor, axis=axis, keepdims=keepdims)


reduce_prod.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reduce_std(input_tensor, axis=None, keepdims=False, name="reduce_std"):
    return ivy.std(input_tensor, axis=axis, keepdims=keepdims)


reduce_std.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64", "float16", \
                    "float32", "float64"),
    "torch": ("float16", "bfloat16"),
    "numpy": ("float16", "bfloat16", "float32", "float64")
}
