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


def _batch_promotion(*args, default_dtype="float64"):
    # Promote all types

    promote_types = set()

    for arg in args:
        if args is None:
            continue
        if isinstance(arg, float) or isinstance(arg, int):
            continue
        promote_types.add(str(arg.dtype))

    if "float64" in promote_types:
        return "float64"

    if "float32" in promote_types:
        return "float32"

    if "float16" in promote_types and "bfloat16" in promote_types:
        return "float32"

    if "float16" in promote_types:
        return "float16"

    if "bfloat16" in promote_types:
        return "bfloat16"

    return default_dtype


def _mean(x, axis=None, keepdims=False, where=None):
    # Mean with support for where
    if where is None:
        return ivy.mean(x, axis=axis, keepdims=keepdims)

    filtered_x = ivy.where(where, ivy.array(x), ivy.zeros_like(x))
    counter_x = ivy.where(where, ivy.ones_like(x), ivy.zeros_like(x))

    sums = ivy.sum(filtered_x, axis=axis, keepdims=keepdims)
    counts = ivy.sum(counter_x, axis=axis, keepdims=keepdims)

    return ivy.divide(sums, counts)


def relu(x):
    return ivy.relu(x)


relu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


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


def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5, where=None):
    default = "float64" if mean is not None and variance is not None else "float32"

    x_typed = _type_conversion(x)
    if mean is None:
        mean = _mean(x_typed, axis=axis, keepdims=True, where=where)
    if variance is None:
        variance = _mean(
            ivy.square(x).astype(x_typed.dtype), axis=axis, keepdims=True, where=where
        ) - ivy.square(mean)

    res = (x - mean) / ivy.sqrt(variance + ivy.asarray(epsilon, dtype=x_typed.dtype))

    out_type = _batch_promotion(x, mean, variance, default_dtype=default)

    return ivy.asarray(res, dtype=out_type)


normalize.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def hard_tanh(x):
    x = ivy.asarray(x)
    n1 = -1
    if "uint" in str(x.dtype):
        dtype = x.dtype
        # tensorflow can't use -1 for uint
        n1 = ivy.asarray((1 << ivy.dtype_bits(dtype)) - 1, dtype=dtype)

    return ivy.where(x > 1, 1, ivy.where(x < n1, n1, x))


hard_tanh.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def _celu_result_dtype(x, alpha):
    x_native = isinstance(x, int) or isinstance(x, float)
    alpha_native = isinstance(alpha, int) or isinstance(alpha, float)
    if x_native and alpha_native:
        return "float64"

    if x_native:
        return _type_conversion(alpha).dtype

    if alpha_native:
        return _type_conversion(x).dtype

    dtypes = [str(ivy.dtype(x)), str(ivy.dtype(alpha))]

    if "float64" in dtypes:
        return "float64"
    if "float32" in dtypes:
        return "float32"
    if "bfloat16" in dtypes:
        return "bfloat16"
    if "float16" in dtypes:
        return "float16"
    if "int64" in dtypes or "uint64" in dtypes:
        return "float64"

    if "uint32" in dtypes and any(d in dtypes for d in ["int8", "int16", "int32"]):
        return "float64"

    return "float32"


def celu(x, alpha=1.0):
    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x / alpha))
    dtype = _celu_result_dtype(x, alpha)
    return ivy.asarray(ret, dtype=dtype)


celu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
