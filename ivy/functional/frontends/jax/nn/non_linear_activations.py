import ivy


# Helpers #
###########
def _type_conversion(x):
    # Does type conversion, floats maps to float,
    # 64bit dtype to float64, everything else to float32
    x = ivy.asarray(x)
    dtype = ivy.as_ivy_dtype(x.dtype)
    if "float" not in dtype:
        if "64" in dtype[-2:]:
            dtype = "float64"
        else:
            dtype = "float32"

    return ivy.astype(x, dtype)


def _type_conversion_64(x):
    # Does type conversion, floats maps to float,
    # everything else to float64
    x = ivy.asarray(x)
    dtype = ivy.as_ivy_dtype(x.dtype)
    if "float" in dtype:
        return ivy.astype(x, dtype)

    return ivy.astype(x, "float64")


def relu(x):
    return ivy.relu(x)


relu.unsupported_dtypes = {"torch": ("float16",)}


def relu6(x):
    res = ivy.minimum(ivy.maximum(x, 0.0), 6.0)
    return _type_conversion_64(res)


relu6.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def soft_sign(x):
    dtype = _type_conversion(x).dtype
    ret = x / (ivy.abs(x) + 1)
    return ret.astype(dtype)


soft_sign.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def silu(x):
    x = _type_conversion(x)
    return x * sigmoid(x)


silu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def leaky_relu(x, negative_slope=0.01):
    x = _type_conversion_64(x)
    return ivy.leaky_relu(x, alpha=negative_slope)


leaky_relu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def gelu(x, approximate=True):
    return ivy.gelu(x, approximate=approximate)


gelu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sigmoid(x):
    x = _type_conversion(x)
    ret = ivy.sigmoid(x)
    return ivy.astype(ret, x.dtype)


sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def one_hot(x, num_classes, *, device=None, out=None):
    ret = ivy.one_hot(x, num_classes, device=device, out=out)
    return ret.astype("float64")


one_hot.supported_dtypes = {"tensorflow": ("uint8", "int32", "int64")}


def softmax(x, /, *, axis=-1):
    dtype = _type_conversion(x).dtype
    ret = ivy.softmax(x, axis=axis)
    return ret.astype(dtype)


softmax.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def softplus(x):
    x = _type_conversion(x)
    return ivy.softplus(x).astype(x.dtype)


softplus.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def log_sigmoid(x):
    x = _type_conversion(x)
    return -ivy.softplus(-x).astype(x.dtype)


log_sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def log_softmax(x, axis=-1):
    x_max = ivy.max(x)
    shifted = ivy.subtract(x, x_max)
    shifted_logsumexp = ivy.log(ivy.sum(ivy.exp(shifted), axis=axis, keepdims=True))
    ret = shifted - shifted_logsumexp
    return ret


log_softmax.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def glu(x, axis=-1):
    size = x.shape[axis]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = ivy.split(x, num_or_size_splits=2, axis=axis)
    return ivy.multiply(x1, ivy.sigmoid(x2))


glu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
