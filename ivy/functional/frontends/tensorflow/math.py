# global
from cmath import nan
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


def reciprocal_no_nan(input_tensor, name="reciprocal_no_nan"):
    return input_tensor.map(lambda x: 1.0 / x if (x != 0) or (x != nan) else 0.0)


def reduce_all(input_tensor, axis=None, keepdims=False, name="reduce_all"):
    return ivy.all(input_tensor, axis, keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False, name="reduce_any"):
    return ivy.any(input_tensor, axis, keepdims)


def reduce_euclidean_norm(
    input_tensor, axis=None, keepdims=False, name="reduce_euclidean_norm"
):
    return ivy.vector_norm(
        input_tensor, axis, keepdims, ord="2"
    )  # ord = '2' is the euclidean norm


def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
    return ivy.exp(input_tensor).sum(axis, keepdims).log()
