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
        promote_types.add(ivy.dtype(arg))

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

    if "int64" in promote_types or "uint64" in promote_types:
        return "float64"

    ints = ["int8", "int16", "int32"]
    if "uint32" in promote_types and any(d in promote_types for d in ints):
        return "float64"

    return default_dtype


def _canonicalize_axis(axis, ndim):
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis = axis + ndim
    return axis


def _len(x):
    shape = ivy.shape(x)
    if len(shape) == 0:
        return 0
    return shape[0]


def _reduction_dims(a, axis):
    ndims = len(ivy.shape(a))
    if axis is None:
        return (tuple(range(ndims)),) * 2
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    canon_axis = tuple(_canonicalize_axis(ax, ndims) for ax in axis)
    if len(canon_axis) != len(set(canon_axis)):
        raise ValueError(f"duplicate value in 'axis': {axis}")

    # TODO: deal with named axis

    canon_pos_axis = tuple(x for x in canon_axis if isinstance(x, int))

    if len(canon_pos_axis) != len(canon_axis):
        return canon_pos_axis, canon_axis
    else:
        return canon_axis, canon_axis


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


def relu6(x):
    res = ivy.minimum(ivy.maximum(x, 0.0), 6.0)
    return _type_conversion_64(res)


def soft_sign(x):
    dtype = _type_conversion(x).dtype
    ret = x / (ivy.abs(x) + 1)
    return ret.astype(dtype)


def silu(x):
    x = _type_conversion(x)
    return x * sigmoid(x)


def leaky_relu(x, negative_slope=0.01):
    x = _type_conversion_64(x)
    return ivy.leaky_relu(x, alpha=negative_slope)


def gelu(x, approximate=True):
    return ivy.gelu(x, approximate=approximate)


def sigmoid(x):
    x = _type_conversion(x)
    ret = ivy.sigmoid(x)
    return ivy.astype(ret, x.dtype)


def one_hot(x, num_classes, *, device=None, out=None):
    ret = ivy.one_hot(x, num_classes, device=device, out=out)
    return ret.astype("float64")


def softmax(x, /, *, axis=-1):
    dtype = _type_conversion(x).dtype
    ret = ivy.softmax(x, axis=axis)
    return ret.astype(dtype)


def softplus(x):
    x = _type_conversion(x)
    return ivy.softplus(x).astype(x.dtype)


def log_sigmoid(x):
    x = _type_conversion(x)
    return -ivy.softplus(-x).astype(x.dtype)


def log_softmax(x, axis=-1):
    x_max = ivy.max(x)
    shifted = ivy.subtract(x, x_max)
    shifted_logsumexp = ivy.log(ivy.sum(ivy.exp(shifted), axis=axis, keepdims=True))
    return shifted - shifted_logsumexp


def glu(x, axis=-1):
    size = x.shape[axis]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = ivy.split(x, num_or_size_splits=2, axis=axis)
    return ivy.multiply(x1, ivy.sigmoid(x2))


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


def hard_tanh(x):
    x = ivy.asarray(x)
    n1 = -1
    if "uint" in str(x.dtype):
        dtype = x.dtype
        # tensorflow can't use -1 for uint
        n1 = ivy.asarray((1 << ivy.dtype_bits(dtype)) - 1, dtype=dtype)

    return ivy.where(x > 1, 1, ivy.where(x < n1, n1, x))


def celu(x, alpha=1.0):
    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x / alpha))
    dtype = _batch_promotion(x, alpha, default_dtype="float32")
    return ivy.asarray(ret, dtype=dtype)


def elu(x, alpha=1.0):
    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x))
    dtype = _batch_promotion(x, alpha, default_dtype="float64")
    return ivy.asarray(ret, dtype=dtype)


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    a = ivy.asarray(a)
    if b is not None:
        dtype = _batch_promotion(a, b, default_dtype="float32")
        a = ivy.astype(a, dtype)
        b = ivy.asarray(b, dtype=dtype)
        a = ivy.where(b != 0, a, -ivy.inf)

    out_dtype = _batch_promotion(a, b, default_dtype="float32")

    pos_dims, dims = _reduction_dims(a, axis)

    amax = ivy.max(a, axis=pos_dims, keepdims=keepdims)
    notinf = ivy.asarray(not ivy.isinf(amax))
    amax = ivy.stop_gradient(ivy.where(notinf, amax, ivy.zeros_like(amax)))
    amax_with_dims = amax if keepdims else ivy.expand_dims(amax, axis=pos_dims)

    # fast path for non-negative result
    if b is None:
        out = ivy.add(
            ivy.log(
                ivy.sum(
                    ivy.exp(ivy.subtract(a, amax_with_dims)),
                    axis=dims,
                    keepdims=keepdims,
                )
            ),
            amax,
        )
        sign = ivy.where(ivy.isnan(out), out, 1.0)
        sign = ivy.where(ivy.isinf(-out), 0.0, sign).astype(out.dtype)
    else:
        expsub = ivy.exp(ivy.subtract(a, amax_with_dims))
        if b is not None:
            expsub = ivy.multiply(expsub, b)
        sumexp = ivy.sum(expsub, axis=dims, keepdims=keepdims)
        sign = ivy.stop_gradient(ivy.sign(sumexp))
        out = ivy.add(ivy.log(ivy.abs(sumexp)), amax)

    if return_sign:
        return out, sign

    if b is not None:
        out = ivy.where(sign < 0, ivy.array(ivy.nan, dtype=out.dtype), out)

    return out.astype(out_dtype)


def swish(x):
    ret = x / (1 + ivy.exp(-x))
    return ivy.asarray(ret, dtype=x.dtype)


def hard_swish(x):
    res = (x * ivy.minimum(ivy.maximum(x + 3, 0.0), 6.0)) / 6
    return ivy.asarray(res, dtype=x.dtype)
