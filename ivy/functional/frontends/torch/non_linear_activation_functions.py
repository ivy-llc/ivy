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


def sigmoid(input):
    return ivy.sigmoid(input)


def leaky_relu(input, negative_slope=0.01):
    return ivy.leaky_relu(input, alpha=negative_slope)


def softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


def gelu(
    input,
):  # , *, approximate="none"): ToDo: approximate is added in in PyTorch 1.12.1
    # if approximate == "none":
    # approximate = False
    # else:
    # approximate = True
    return ivy.gelu(input)


def tanh(input):
    return ivy.tanh(input)


def logsigmoid(input):
    return -ivy.softplus(-input)


def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value, inplace)


def threshold_(input, threshold, value):
    return _compute_threshold(input, threshold, value, inplace=True)


def relu6(input, inplace=False):
    if inplace:
        return ivy.minimum(ivy.maximum(input, 0), 6, out=input)
    return ivy.minimum(ivy.maximum(input, 0), 6)


def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha, inplace=inplace)


def elu_(input, alpha=1.0):
    return _compute_elu(input, alpha, inplace=True)


def celu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(ivy.divide(input, alpha)),
            1,
        ),
    )
    ret = ivy.add(
        ivy.maximum(0, input),
        ivy.minimum(0, prod),
    )
    if inplace:
        return ivy.inplace_update(input, ret)
    return ret


celu.unsupported_dtypes = ("float16",)
