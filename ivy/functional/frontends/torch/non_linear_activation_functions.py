# global
import ivy


def _compute_threshold(input, threshold, value, inplace):
    ret = ivy.where(ivy.greater(input, threshold), input, value)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _compute_elu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    ret = ivy.where(ivy.greater(input, 0), input, prod)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _selu_with_inplace(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(input),
            1,
        ),
    )
    min_ = ivy.multiply(
        scale,
        ivy.minimum(0, prod),
    )
    max_ = ivy.multiply(
        scale,
        ivy.maximum(0, input),
    )
    ret = ivy.add(min_, max_)
    if inplace:
        return ivy.inplace_update(input, ret)
    return ret


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
    return ivy.log(ivy.sigmoid(input))


def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


softmin.unsupported_dtypes = ("float16",)


def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value, inplace)


def threshold_(input, threshold, value):
    return _compute_threshold(input, threshold, value, inplace=True)


def relu6(input, inplace=False):
    ret = ivy.minimum(ivy.maximum(input, 0), 6)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


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
        ivy.inplace_update(input, ret)
        return input
    return ret


celu.unsupported_dtypes = ("float16",)


def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    import torch
    assert ivy.all(ivy.greater(max_val,min_val))
    if inplace:
        result = torch._C._nn.hardtanh_(input, min_val, max_val)
    else:
        if ivy.all(ivy.greater(input, max_val)):
            result = max_val
        if ivy.all(ivy.less(input, min_val)):
            result = min_val
        if ivy.all(ivy.less_equal(input, max_val)
                   and ivy.greater_equal(input, min_val)):
            result = input

    return result


hardtanh.unsupported_dtypes = ("float16",)


def selu(input, inplace=False):
    return _selu_with_inplace(input, inplace=inplace)
