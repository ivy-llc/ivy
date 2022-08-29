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


logical_xor.supported_dtypes = {"torch": ("bool",)}


def divide(x, y, name=None):
    return ivy.divide(x, y)


def negative(x, name=None):
    return ivy.negative(x)


negative.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reciprocal_no_nan(input_tensor, name="reciprocal_no_nan"):
    return ivy.where(
        input_tensor == 0,
        ivy.array(0.0, dtype=input_tensor.dtype),
        ivy.ones_like(input_tensor, dtype=input_tensor.dtype) / input_tensor,
    )


def reduce_all(input_tensor, axis=None, keepdims=False, name="reduce_all"):
    return ivy.all(input_tensor, axis=axis, keepdims=keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False, name="reduce_any"):
    return ivy.any(input_tensor, axis=axis, keepdims=keepdims)


def reduce_euclidean_norm(
    input_tensor, axis=None, keepdims=False, name="reduce_euclidean_norm"
):
    return ivy.vector_norm(
        input_tensor, axis=axis, keepdims=keepdims, ord=2
    )  # ord = '2' is the euclidean norm


reduce_euclidean_norm.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
    return ivy.exp(input_tensor).sum(axis=axis, keepdims=keepdims).log()


reduce_logsumexp.unsupported_dtypes = {
    "tensorflow": (
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ),
    "torch": ("float16",),
}


def logical_and(x, y, name="LogicalAnd"):
    return ivy.logical_and(x, y)


logical_and.supported_dtypes = ("bool",)


def argmax(input, axis, output_type, name=None):
    return ivy.argmax(input, axis=axis)


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
    "tensorflow": (
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ),
}


def asinh(x, name="asinh"):
    return ivy.asinh(x)



asinh.unsupported_dtypes = {"torch": ("float16",)}


def reduce_sum(input_tensor, axis=None, keepdims=False, name="reduce_sum"):
    return ivy.sum(input_tensor, axis=axis, keepdims=keepdims)


reduce_sum.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def reduce_variance(input_tensor, axis=None, keepdims=False, name="reduce_variance"):
    return ivy.var(input_tensor, axis=axis, keepdims=keepdims)


reduce_variance.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "uint64"),
}


def scalar_mul(scalar, x, name="scalar_mul"):
    return ivy.multiply(x, ivy.array([scalar]))


scalar_mul.unsupported_dtypes = {
    "tensorflow": (
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ),
    "torch": ("float16", "bfloat16"),
    "numpy": ("float16", "bfloat16", "float32", "float64"),
}

def log_sigmoid(x, name=None):
    return -ivy.softplus(-x)


log_sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
    ret = ivy.cumprod(x, axis, exclusive)
    if reverse:
        return ivy.flip(ret, axis)
    return ret

def divide_no_nan(x, y, name="divide_no_nan"):
    return ivy.where(
        y == 0,
        ivy.array(0.0, dtype=ivy.promote_types(x.dtype, y.dtype)),
        x / y,
    )


def erfcinv(x, name="erfcinv"):
    return 1 / (1 - ivy.erf(x))


def is_non_decreasing(x, name="is_non_decreasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array(x[0] <= x[1])
    return ivy.all(ivy.less_equal(x, ivy.roll(x, -1)))


def is_strictly_increasing(x, name="is_strictly_increasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array(x[0] < x[1])
    return ivy.all(ivy.less(x, ivy.roll(x, -1)))


# TODO: Ibeta for Future Release
