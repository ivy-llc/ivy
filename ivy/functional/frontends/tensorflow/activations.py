import ivy


def hard_sigmoid(x):
    dtype_in = x.dtype
    point_two = ivy.full(x.shape, 0.2)
    point_five = ivy.full(x.shape, 0.5)
    x = ivy.multiply(x, point_two)
    x = ivy.add(x, point_five)
    x = ivy.clip(x, 0.0, 1.0)
    x = ivy.asarray(x, dtype=dtype_in)
    return x


def linear(x):
    return x


def sigmoid(x):
    return ivy.sigmoid(x)


def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)


def elu(x, alpha=1.0):
    zeros = ivy.zeros_like(x)
    ones = ivy.ones_like(x)
    ret_val = ivy.where(
        x > zeros, x, ivy.multiply(alpha, ivy.subtract(ivy.exp(x), ones))
    )
    return ret_val


elu.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "bool",
)


def selu(x):
    alpha = 1.67326324
    scale = 1.05070098
    return ivy.multiply(scale, elu(x=x, alpha=alpha))


selu.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "bool",
)
