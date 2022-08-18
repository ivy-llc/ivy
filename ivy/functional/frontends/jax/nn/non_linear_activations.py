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
    x = _type_conversion(x)
    return x / (ivy.abs(x) + 1)


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
    # TODO: Fix gelu backend implementation
    if approximate:
        x = _type_conversion_64(x)
    ret = ivy.gelu(x, approximate=approximate)
    return ret.astype(x.dtype)


gelu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sigmoid(x):
    x = _type_conversion(x)
    ret = ivy.sigmoid(x)
    return ivy.astype(ret, x.dtype)


sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def one_hot(x, num_classes, *, device=None, out=None):
    return ivy.one_hot(x, num_classes, device=device, out=out)


one_hot.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def softmax(x, /, *, axis=-1):
    x = _type_conversion(x)
    return ivy.softmax(x, axis=axis).astype(x.dtype)


softmax.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def softplus(x):
    x = _type_conversion(x)
    return ivy.softplus(x).astype(x.dtype)


softplus.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5, where=None):
    # TODO: implement where in mean
    if mean is None:
        mean = ivy.mean(x, axis=axis, where=where)
    if variance is None:
        variance = ivy.mean(
            ivy.square(x, axis, keepdims=True), axis=axis, where=where
        ) - ivy.square(mean)

    return (x - mean) * ivy.sqrt(variance + epsilon)


sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
