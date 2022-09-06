# global
import ivy


def _compute_threshold(input, threshold, value, inplace):
    if inplace:
        return ivy.where(ivy.greater(input, threshold), input, value, out=input)
    return ivy.where(ivy.greater(input, threshold), input, value)


def _compute_elu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    if inplace:
        input = ivy.where(ivy.greater(input, 0), input, prod)
        return input
    return ivy.where(ivy.greater(input, 0), input, prod)


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


def leaky_relu(input, negative_slope=0.01):
    return ivy.leaky_relu(input, alpha=negative_slope)


leaky_relu.unsupported_dtypes = ("float16",)


def softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


softmax.unsupported_dtypes = ("float16",)


def gelu(input, approximate="none"):
    if approximate == "none":
        approximate = False
    else:
        approximate = True
    return ivy.gelu(input, approximate)


gelu.unsupported_dtypes = ("float16",)


def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


tanh.unsupported_dtypes = {"torch": ("float16",)}


def logsigmoid(input):
    return -ivy.softplus(-input)


logsigmoid.unsupported_dtypes = ("float16",)


def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


softmin.unsupported_dtypes = ("float16",)


def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value, inplace)


threshold.unsupported_dtypes = ("float16",)


def threshold_(input, threshold, value):
    return _compute_threshold(input, threshold, value, inplace=True)


threshold_.unsupported_dtypes = ("float16",)


def relu6(input, inplace=False):
    if inplace:
        return ivy.minimum(ivy.maximum(input, 0), 6, out=input)
    return ivy.minimum(ivy.maximum(input, 0), 6)


relu6.unsupported_dtypes = ("float16",)


def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha, inplace=inplace)


elu.unsupported_dtypes = ("float16",)


def elu_(input, alpha=1.0):
    return _compute_elu(input, alpha, inplace=True)


elu_.unsupported_dtypes = ("float16",)
